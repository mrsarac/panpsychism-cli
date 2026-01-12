use clap::{Parser, Subcommand};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use panpsychism::gemini::{GeminiClient, GeminiModel};
use panpsychism::indexer::Indexer;
use panpsychism::search::{PromptMetadata, SearchEngine, SearchQuery};
use panpsychism::{setup_logging, should_use_json, Config};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, info_span, instrument, warn, Instrument};
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "panpsychism")]
#[command(author, version, about = "Prompt orchestration with semantic memory", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output logs in JSON format
    #[arg(long, global = true)]
    json: bool,

    /// Set log level (error, warn, info, debug, trace)
    #[arg(long, global = true, default_value = "info")]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Index prompts into .mv2 file
    Index {
        /// Prompts directory
        #[arg(short, long, default_value = "./prompts")]
        dir: PathBuf,

        /// Output .mv2 file
        #[arg(short, long, default_value = "./data/masters.mv2")]
        output: PathBuf,
    },

    /// Search for prompts
    Search {
        /// The search query
        query: String,

        /// Number of results to return
        #[arg(short, long, default_value = "5")]
        top: usize,
    },

    /// Ask a question with orchestrated prompts
    Ask {
        /// The question to ask
        question: String,

        /// Show reasoning trace
        #[arg(short, long)]
        verbose: bool,
    },

    /// Interactive shell mode
    Shell,

    /// Analyze text from SuperWhisper or other sources
    Analyze {
        /// Text to analyze (optional if using --input)
        text: Option<String>,

        /// Input source: "stdin", "clipboard", or file path
        #[arg(short, long)]
        input: Option<String>,

        /// Audio file path (wav/mp3) for transcription
        #[arg(short, long)]
        audio: Option<PathBuf>,

        /// Output destination: "clipboard", "stdout", or file path
        #[arg(short, long, default_value = "stdout")]
        output: String,

        /// Verbose output with processing details
        #[arg(short, long)]
        verbose: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing with configurable format
    let use_json = cli.json || should_use_json();
    setup_logging(use_json, &cli.log_level);

    info!(
        version = env!("CARGO_PKG_VERSION"),
        json_output = use_json,
        "Panpsychism starting"
    );

    match cli.command {
        Commands::Index { dir, output } => {
            cmd_index(dir, output).await?;
        }
        Commands::Search { query, top } => {
            cmd_search(query, top).await?;
        }
        Commands::Ask { question, verbose } => {
            cmd_ask(question, verbose).await?;
        }
        Commands::Shell => {
            cmd_shell().await?;
        }
        Commands::Analyze {
            text,
            input,
            audio,
            output,
            verbose,
        } => {
            cmd_analyze(text, input, audio, output, verbose).await?;
        }
    }

    Ok(())
}

#[instrument(skip_all, fields(prompts_dir = %dir.display(), output_file = %output.display()))]
async fn cmd_index(dir: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    let start = Instant::now();
    let request_id = Uuid::new_v4();
    let span = info_span!("index_request", id = %request_id);

    async move {
        info!("Starting indexing operation");

        // Check if prompts directory exists
        if !dir.exists() {
            error!(path = %dir.display(), "Prompts directory not found");
            eprintln!(
                "{} Directory not found: {}",
                "Error:".red().bold(),
                dir.display().to_string().yellow()
            );
            eprintln!();
            eprintln!(
                "{} Create the prompts directory first:",
                "Hint:".cyan().bold()
            );
            eprintln!("      mkdir -p {}", dir.display());
            eprintln!();
            eprintln!(
                "      Then add prompt files with YAML frontmatter (e.g., {})",
                "my-prompt.md".green()
            );
            return Err(anyhow::anyhow!(
                "Prompts directory not found: {}",
                dir.display()
            ));
        }

        // Create output directory if it doesn't exist
        if let Some(parent) = output.parent() {
            if !parent.exists() {
                info!(path = %parent.display(), "Creating data directory");
                println!(
                    "{} Creating data directory: {}",
                    "Info:".blue().bold(),
                    parent.display().to_string().cyan()
                );
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    error!(error = %e, path = %parent.display(), "Failed to create data directory");
                    anyhow::anyhow!(
                        "Failed to create data directory '{}': {}. Check write permissions.",
                        parent.display(),
                        e
                    )
                })?;
            }
        }

        // Start scanning message
        info!("Scanning prompts directory");
        println!(
            "{} Scanning prompts directory: {}",
            "🔍".cyan(),
            dir.display().to_string().green()
        );

        // Create a spinner for the scanning phase
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
                .template("{spinner:.cyan} {msg}")
                .unwrap(),
        );
        spinner.set_message("Discovering prompt files...");
        spinner.enable_steady_tick(std::time::Duration::from_millis(80));

        // Create indexer and run indexing
        let mut indexer = Indexer::new(&dir, &output);
        let stats = indexer.index().await;

        // Stop the spinner
        spinner.finish_and_clear();

        // Handle result
        match stats {
            Ok(stats) => {
                let duration = start.elapsed();

                info!(
                    prompts_indexed = stats.prompts_indexed,
                    prompts_skipped = stats.prompts_skipped,
                    errors = stats.errors,
                    duration_ms = duration.as_millis() as u64,
                    unique_tags = stats.unique_tags,
                    unique_categories = stats.unique_categories,
                    "Indexing complete"
                );

                // Create progress bar for visual feedback (already complete)
                let total = stats.prompts_indexed + stats.prompts_skipped + stats.errors;
                if total > 0 {
                    let pb = ProgressBar::new(total as u64);
                    pb.set_style(
                        ProgressStyle::default_bar()
                            .template("{bar:40.green/dim} {percent}% [{pos}/{len}]")
                            .unwrap()
                            .progress_chars("█▓░"),
                    );
                    pb.set_position(total as u64);
                    pb.finish();
                }

                println!();
                println!("{} {}", "✅".green(), "Indexing complete!".green().bold());
                println!(
                    "   {} Directory: {}",
                    "📁".blue(),
                    dir.display().to_string().white()
                );
                println!(
                    "   {} Indexed: {} prompts",
                    "📄".green(),
                    stats.prompts_indexed.to_string().green().bold()
                );

                if stats.prompts_skipped > 0 {
                    warn!(
                        count = stats.prompts_skipped,
                        "Files skipped (invalid format)"
                    );
                    println!(
                        "   {} Skipped: {} files (invalid format)",
                        "⏭️".yellow(),
                        stats.prompts_skipped.to_string().yellow()
                    );
                } else {
                    println!("   {} Skipped: {} files", "⏭️".dimmed(), "0".dimmed());
                }

                if stats.errors > 0 {
                    error!(count = stats.errors, "Indexing errors occurred");
                    println!(
                        "   {} Errors: {}",
                        "❌".red(),
                        stats.errors.to_string().red().bold()
                    );
                } else {
                    println!("   {} Errors: {}", "❌".dimmed(), "0".dimmed());
                }

                println!(
                    "   {} Output: {}",
                    "💾".blue(),
                    output.display().to_string().cyan()
                );
                println!(
                    "   {} Duration: {:.2}s",
                    "⏱️".blue(),
                    duration.as_secs_f64()
                );

                // Additional stats if available
                if stats.unique_tags > 0 || stats.unique_categories > 0 {
                    println!();
                    println!(
                        "   {} {} unique tags, {} categories",
                        "🏷️".magenta(),
                        stats.unique_tags.to_string().magenta(),
                        stats.unique_categories.to_string().magenta()
                    );
                }

                // Warning if no prompts found
                if stats.prompts_indexed == 0 {
                    warn!("No prompts were indexed");
                    println!();
                    println!("{} No prompts were indexed!", "⚠️ Warning:".yellow().bold());
                    println!("         Make sure your .md files have YAML frontmatter:");
                    println!();
                    println!("         {}", "---".dimmed());
                    println!("         {}", "title: My Prompt".dimmed());
                    println!("         {}", "description: A helpful prompt".dimmed());
                    println!("         {}", "tags: [example, demo]".dimmed());
                    println!("         {}", "category: general".dimmed());
                    println!("         {}", "---".dimmed());
                }

                Ok(())
            }
            Err(e) => {
                error!(error = %e, "Indexing failed");
                eprintln!("{} {}", "❌ Error:".red().bold(), e.to_string().red());

                // Provide helpful error messages based on error type
                let error_str = e.to_string();
                if error_str.contains("permission") || error_str.contains("Permission") {
                    eprintln!();
                    eprintln!(
                        "{} Check file and directory permissions:",
                        "💡 Hint:".cyan().bold()
                    );
                    eprintln!("      ls -la {}", dir.display());
                    eprintln!(
                        "      ls -la {}",
                        output.parent().unwrap_or(&output).display()
                    );
                }

                Err(anyhow::anyhow!("Indexing failed: {}", e))
            }
        }
    }
    .instrument(span)
    .await
}

/// Analyze text from SuperWhisper or other sources.
///
/// This command reads text from various sources (direct argument, stdin, file, or clipboard),
/// and outputs the result to stdout, a file, or clipboard.
///
/// # Arguments
/// * `text` - Direct text input (optional)
/// * `input` - Input source: "stdin", "clipboard", or file path
/// * `audio` - Audio file path for transcription (placeholder)
/// * `output` - Output destination: "stdout", "clipboard", or file path
/// * `verbose` - Show processing details
#[instrument(skip_all, fields(has_text = text.is_some(), has_input = input.is_some(), has_audio = audio.is_some()))]
async fn cmd_analyze(
    text: Option<String>,
    input: Option<String>,
    audio: Option<PathBuf>,
    output: String,
    verbose: bool,
) -> anyhow::Result<()> {
    let start = Instant::now();
    let request_id = Uuid::new_v4();
    let span = info_span!("analyze_request", id = %request_id);

    async move {
        info!("Starting analyze operation");

        if verbose {
            println!();
            println!("{}", "Analyze Command".cyan().bold());
            println!("{}", "=".repeat(40).dimmed());
        }

        // Step 1: Read input from the appropriate source
        let input_text = read_analyze_input(text, input.clone(), audio.clone(), verbose).await?;

        if input_text.is_empty() {
            error!("No input text provided");
            eprintln!(
                "{} No input text provided. Use positional argument, --input, or --audio.",
                "Error:".red().bold()
            );
            return Err(anyhow::anyhow!("No input text provided"));
        }

        if verbose {
            println!();
            println!("{} Input received ({} chars)", "✓".green(), input_text.len());
            println!("{}", "─".repeat(40).dimmed());
            // Show first 200 chars as preview
            let preview = if input_text.len() > 200 {
                format!("{}...", &input_text[..200])
            } else {
                input_text.clone()
            };
            println!("{}", preview.dimmed());
            println!("{}", "─".repeat(40).dimmed());
        }

        // Step 2: Process the text (placeholder - just echo for now)
        // TODO: Add actual processing logic (LLM analysis, intent detection, etc.)
        let processed_text = input_text.clone();

        // Step 3: Write output to the appropriate destination
        write_analyze_output(&processed_text, &output, verbose).await?;

        let duration = start.elapsed();
        info!(
            duration_ms = duration.as_millis() as u64,
            input_chars = input_text.len(),
            output_chars = processed_text.len(),
            "Analyze operation complete"
        );

        if verbose {
            println!();
            println!(
                "{} Completed in {:.2}s",
                "✓".green(),
                duration.as_secs_f64()
            );
        }

        Ok(())
    }
    .instrument(span)
    .await
}

/// Read input from the appropriate source for analyze command.
async fn read_analyze_input(
    text: Option<String>,
    input: Option<String>,
    audio: Option<PathBuf>,
    verbose: bool,
) -> anyhow::Result<String> {
    // Priority: text argument > audio > input source

    // 1. Direct text argument
    if let Some(t) = text {
        debug!("Using direct text argument");
        if verbose {
            println!("{} Reading from: direct argument", "→".blue());
        }
        return Ok(t);
    }

    // 2. Audio file (placeholder - return error for now)
    if let Some(audio_path) = audio {
        debug!(path = %audio_path.display(), "Audio file specified");
        if verbose {
            println!("{} Reading from: audio file ({})", "→".blue(), audio_path.display());
        }
        // TODO: Implement audio transcription using Whisper or similar
        return Err(anyhow::anyhow!(
            "Audio transcription not yet implemented. File: {}",
            audio_path.display()
        ));
    }

    // 3. Input source (stdin, clipboard, or file)
    if let Some(source) = input {
        match source.as_str() {
            "stdin" => {
                debug!("Reading from stdin");
                if verbose {
                    println!("{} Reading from: stdin", "→".blue());
                }
                use std::io::{self, Read};
                let mut buffer = String::new();
                io::stdin().read_to_string(&mut buffer)?;
                return Ok(buffer.trim().to_string());
            }
            "clipboard" => {
                debug!("Reading from clipboard");
                if verbose {
                    println!("{} Reading from: clipboard", "→".blue());
                }
                // TODO: Implement clipboard reading (requires additional dependency)
                return Err(anyhow::anyhow!(
                    "Clipboard support not yet implemented. Use --input stdin or provide a file path."
                ));
            }
            file_path => {
                debug!(path = file_path, "Reading from file");
                if verbose {
                    println!("{} Reading from: file ({})", "→".blue(), file_path);
                }
                let path = PathBuf::from(file_path);
                if !path.exists() {
                    return Err(anyhow::anyhow!("Input file not found: {}", file_path));
                }
                let content = tokio::fs::read_to_string(&path).await?;
                return Ok(content);
            }
        }
    }

    // No input source specified - return empty
    Ok(String::new())
}

/// Write output to the appropriate destination for analyze command.
async fn write_analyze_output(text: &str, output: &str, verbose: bool) -> anyhow::Result<()> {
    match output {
        "stdout" => {
            debug!("Writing to stdout");
            if verbose {
                println!("{} Writing to: stdout", "→".blue());
                println!();
            }
            println!("{}", text);
        }
        "clipboard" => {
            debug!("Writing to clipboard");
            if verbose {
                println!("{} Writing to: clipboard", "→".blue());
            }
            // TODO: Implement clipboard writing (requires additional dependency)
            return Err(anyhow::anyhow!(
                "Clipboard support not yet implemented. Use --output stdout or provide a file path."
            ));
        }
        file_path => {
            debug!(path = file_path, "Writing to file");
            if verbose {
                println!("{} Writing to: file ({})", "→".blue(), file_path);
            }
            tokio::fs::write(file_path, text).await?;
            if verbose {
                println!("{} Saved to {}", "✓".green(), file_path);
            }
        }
    }

    Ok(())
}

#[instrument(skip_all, fields(query = %query, top_k = top))]
async fn cmd_search(query: String, top: usize) -> anyhow::Result<()> {
    let start = Instant::now();
    let request_id = Uuid::new_v4();
    let span = info_span!("search_request", id = %request_id);

    async move {
        info!("Starting search");

        // Print search header
        println!();
        println!("{} Searching for: \"{}\"", "🔍".cyan(), query.cyan().bold());
        println!();

        // Load index from prompts directory
        let prompts_dir = PathBuf::from("./prompts");
        let index_path = PathBuf::from("./data/masters.mv2");

        if !prompts_dir.exists() {
            warn!(path = %prompts_dir.display(), "Prompts directory not found");
            print_no_results(&query, "Prompts directory not found at ./prompts");
            return Ok(());
        }

        let mut indexer = Indexer::new(&prompts_dir, &index_path);

        // Index prompts (in production, this would be cached)
        debug!("Building index");
        if let Err(e) = indexer.index().await {
            error!(error = %e, "Failed to index prompts");
            print_no_results(&query, &format!("Failed to index prompts: {}", e));
            return Ok(());
        }

        let index = indexer.get_index();

        if index.is_empty() {
            warn!("No prompts found in library");
            print_no_results(&query, "No prompts found in the library");
            return Ok(());
        }

        debug!(count = index.len(), "Prompts loaded into search engine");

        // Convert indexer entries to search engine format
        let prompts: Vec<PromptMetadata> = index
            .iter()
            .map(|entry| {
                PromptMetadata::new(
                    entry.path.to_string_lossy().to_string(),
                    &entry.metadata.title,
                    &entry.content,
                    &entry.path,
                )
                .with_tags(entry.metadata.tags.clone())
                .with_category(&entry.metadata.category)
            })
            .collect();

        // Create search engine and perform search
        let engine = SearchEngine::new(prompts);
        let search_query = SearchQuery::new(&query).with_top_k(top);

        let results = match engine.search(&search_query).await {
            Ok(results) => results,
            Err(e) => {
                error!(error = %e, "Search failed");
                print_no_results(&query, &format!("Search error: {}", e));
                return Ok(());
            }
        };

        // Filter out zero-score results
        let results: Vec<_> = results.into_iter().filter(|r| r.score > 0.0).collect();

        if results.is_empty() {
            info!("No results found");
            print_no_results(&query, "");
            return Ok(());
        }

        info!(count = results.len(), "Search results found");

        // Display results
        for (i, result) in results.iter().enumerate() {
            debug!(
                rank = i + 1,
                title = %result.title,
                score = result.score,
                "Search result"
            );
            print_result_box(i + 1, result, &query);
        }

        // Print footer
        let elapsed = start.elapsed();
        info!(
            count = results.len(),
            duration_ms = elapsed.as_millis() as u64,
            "Search complete"
        );
        println!(
            "{} {} {} in {:.2}s",
            "Found".dimmed(),
            results.len().to_string().green().bold(),
            if results.len() == 1 {
                "result"
            } else {
                "results"
            }
            .dimmed(),
            elapsed.as_secs_f64()
        );
        println!();

        Ok(())
    }
    .instrument(span)
    .await
}

/// Print a search result in a styled box format
fn print_result_box(rank: usize, result: &panpsychism::search::SearchResult, query: &str) {
    let box_width = 65;
    let score_percent = (result.score * 100.0).min(100.0);

    // Format score with color based on value
    let score_str = format!("[{:.1}%]", score_percent);
    let colored_score = if score_percent >= 80.0 {
        score_str.green().bold()
    } else if score_percent >= 50.0 {
        score_str.yellow()
    } else {
        score_str.red()
    };

    // Calculate title line with proper spacing
    let title_prefix = format!("{:>2}. ", rank);

    // Calculate padding for score alignment
    let available_width = box_width - 4; // Account for borders and padding
    let title_display_len = title_prefix.len() + result.title.chars().count();
    let score_display_len = score_str.len();
    let padding_needed = if title_display_len + score_display_len + 1 < available_width {
        available_width - title_display_len - score_display_len
    } else {
        1
    };

    // Top border
    println!("{}", format!("┌{}┐", "─".repeat(box_width)).cyan());

    // Title line with score
    let title_truncated = if result.title.chars().count() > 45 {
        format!("{}...", result.title.chars().take(42).collect::<String>())
    } else {
        result.title.clone()
    };

    println!(
        "{} {}{}{}{} {}",
        "│".cyan(),
        title_prefix.white().bold(),
        title_truncated.white().bold(),
        " ".repeat(
            padding_needed.saturating_sub(
                title_truncated.chars().count()
                    - result
                        .title
                        .chars()
                        .count()
                        .min(title_truncated.chars().count())
            )
        ),
        colored_score,
        "│".cyan()
    );

    // Separator
    println!("{}", format!("├{}┤", "─".repeat(box_width)).cyan());

    // Excerpt with highlighting
    let excerpt = highlight_terms(&result.excerpt, query);
    let excerpt_lines = wrap_text(&excerpt, box_width - 4);
    for line in excerpt_lines.iter().take(2) {
        println!(
            "{} {:<width$} {}",
            "│".cyan(),
            line,
            "│".cyan(),
            width = box_width - 2
        );
    }

    // Tags line
    if !result.tags.is_empty() {
        let tags_str = result
            .tags
            .iter()
            .map(|t| t.cyan().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let tags_display = format!("Tags: {}", tags_str);
        println!(
            "{} {:<width$} {}",
            "│".cyan(),
            tags_display,
            "│".cyan(),
            width = box_width - 2
        );
    }

    // Bottom border
    println!("{}", format!("└{}┘", "─".repeat(box_width)).cyan());
    println!();
}

/// Highlight search terms in text
fn highlight_terms(text: &str, query: &str) -> String {
    let mut result = text.to_string();
    let terms: Vec<&str> = query.split_whitespace().collect();

    for term in terms {
        let term_lower = term.to_lowercase();
        // Simple case-insensitive replacement with highlighting
        let text_lower = result.to_lowercase();
        if let Some(pos) = text_lower.find(&term_lower) {
            let matched_text = &result[pos..pos + term.len()];
            let highlighted = matched_text.yellow().bold().to_string();
            result = format!(
                "{}{}{}",
                &result[..pos],
                highlighted,
                &result[pos + term.len()..]
            );
        }
    }

    result
}

/// Wrap text to fit within a given width
fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    // Remove ANSI codes for length calculation but preserve them in output
    let clean_text = text.replace('\n', " ");

    for word in clean_text.split_whitespace() {
        let word_len = strip_ansi_codes(word).len();
        let current_len = strip_ansi_codes(&current_line).len();

        if current_len + word_len + 1 > max_width && !current_line.is_empty() {
            lines.push(current_line);
            current_line = word.to_string();
        } else if current_line.is_empty() {
            current_line = word.to_string();
        } else {
            current_line.push(' ');
            current_line.push_str(word);
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

/// Strip ANSI escape codes for accurate length calculation
fn strip_ansi_codes(s: &str) -> String {
    let re = regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    re.replace_all(s, "").to_string()
}

/// Print a "no results" message with suggestions
fn print_no_results(query: &str, extra_msg: &str) {
    println!(
        "{} No results found for \"{}\"",
        "😕".yellow(),
        query.yellow()
    );

    if !extra_msg.is_empty() {
        println!("   {}", extra_msg.dimmed());
    }

    println!();
    println!("{}", "💡 Suggestions:".cyan());
    println!("   {} Try broader terms", "•".dimmed());
    println!("   {} Check spelling", "•".dimmed());
    println!("   {} Use fewer keywords", "•".dimmed());
    println!();
}

/// The main intelligence pipeline - Ask command implementation.
#[instrument(skip_all, fields(question_len = question.len(), verbose = verbose))]
async fn cmd_ask(question: String, verbose: bool) -> anyhow::Result<()> {
    use panpsychism::corrector::{Answer, Corrector};
    use panpsychism::gemini::GeminiModel;
    use panpsychism::orchestrator::{Orchestrator, Strategy};
    use panpsychism::synthesizer::TokenUsage;
    use panpsychism::validator::SpinozaValidator;

    let pipeline_start = Instant::now();
    let request_id = Uuid::new_v4();
    let span = info_span!("ask_request", id = %request_id);

    async move {
        info!(question = %question, "Starting ask pipeline");

        // =========================================================================
        // Step 0: Load configuration and initialize components
        // =========================================================================

        if verbose {
            println!();
            println!("{}", "Initializing pipeline...".cyan().bold());
        }

        // Load config (uses defaults if no config file exists)
        let config = Config::load()?;
        debug!(
            prompts_dir = %config.prompts_dir.display(),
            model = %config.llm_model,
            "Configuration loaded"
        );

        // Try to get API key from environment, fall back to config
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("ANTIGRAVITY_API_KEY"))
            .unwrap_or_else(|_| config.llm_api_key.clone());

        // Initialize Gemini client (strip /v1/chat/completions from endpoint if present)
        let endpoint = config.llm_endpoint.replace("/v1/chat/completions", "");
        // Parse model from config string or use default
        let model = match config.llm_model.as_str() {
            "gemini-2.0-flash" => GeminiModel::Flash,
            "gemini-pro" => GeminiModel::Pro,
            "gemini-2.0-flash-thinking" => GeminiModel::ProThinking,
            "gemini-3-flash" => GeminiModel::Flash3,
            "gemini-3-pro-high" => GeminiModel::Pro3High,
            _ => GeminiModel::default(), // Flash3 is default
        };

        let gemini = GeminiClient::new()
            .with_endpoint(&endpoint)
            .with_api_key(&api_key)
            .with_model(model);

        // Initialize other components
        let orchestrator = Orchestrator::new();
        let validator = SpinozaValidator::new();
        let corrector = Corrector::new();

        // Load prompts from index
        let mut indexer = Indexer::new(&config.prompts_dir, &config.index_file);

        // Check if we need to index first
        if !indexer.is_index_valid() {
            debug!(path = %config.prompts_dir.display(), "Building index");
            if verbose {
                println!(
                    "   {} Building index from {:?}...",
                    "".yellow(),
                    config.prompts_dir
                );
            }
            let _ = indexer.index().await; // Ignore errors, we'll work with what we have
        }

        // Convert indexed entries to search metadata
        let prompts: Vec<PromptMetadata> = indexer
            .get_index()
            .iter()
            .map(|entry| {
                PromptMetadata::new(
                    entry.path.to_string_lossy().to_string(),
                    &entry.metadata.title,
                    &entry.content,
                    &entry.path,
                )
                .with_tags(entry.metadata.tags.clone())
                .with_category(&entry.metadata.category)
            })
            .collect();

        let search_engine = SearchEngine::new(prompts.clone());

        info!(prompt_count = prompts.len(), "Prompts loaded into search engine");
        if verbose {
            println!(
                "   {} Loaded {} prompts into search engine",
                "".green(),
                prompts.len()
            );
            println!();
        }

        // =========================================================================
        // Step 1: Search for relevant prompts
        // =========================================================================

        let search_span = info_span!("search_prompts");
        let search_results = async {
            info!("Searching for relevant prompts");
            if verbose {
                println!(
                    "{}",
                    "Step 1: Searching for relevant prompts...".cyan().bold()
                );
            }

            let search_query = SearchQuery::new(&question).with_top_k(10);
            let results = search_engine
                .search(&search_query)
                .await
                .unwrap_or_default();

            info!(count = results.len(), "Search results found");
            if verbose {
                println!(
                    "   Found {} matching prompts",
                    results.len().to_string().green()
                );
                for (i, result) in results.iter().take(5).enumerate() {
                    println!(
                        "   {}. {} (score: {:.2})",
                        i + 1,
                        result.title,
                        result.score
                    );
                }
                println!();
            }
            results
        }
        .instrument(search_span)
        .await;

        // =========================================================================
        // Step 2: Analyze user intent
        // =========================================================================

        let intent_span = info_span!("analyze_intent");
        let intent = async {
            info!("Analyzing user intent");
            if verbose {
                println!("{}", "Step 2: Analyzing intent...".cyan().bold());
            }

            let intent = orchestrator
                .analyze_intent(&question)
                .await
                .map_err(|e| anyhow::anyhow!("Intent analysis failed: {}", e))?;

            info!(
                category = %intent.category,
                complexity = intent.complexity,
                keywords = ?intent.keywords,
                "Intent analyzed"
            );

            if verbose {
                println!("   Category: {}", intent.category.yellow());
                println!("   Complexity: {}/10", intent.complexity);
                println!("   Keywords: {}", intent.keywords.join(", ").dimmed());
                println!();
            }
            Ok::<_, anyhow::Error>(intent)
        }
        .instrument(intent_span)
        .await?;

        // =========================================================================
        // Step 3: Determine strategy
        // =========================================================================

        let strategy_span = info_span!("determine_strategy");
        let strategy = async {
            info!("Determining orchestration strategy");
            if verbose {
                println!("{}", "Step 3: Determining strategy...".cyan().bold());
            }

            let strategy = orchestrator.determine_strategy(&intent, search_results.len());

            let strategy_desc = match strategy {
                Strategy::Focused => "Focused (single expert prompt)",
                Strategy::Ensemble => "Ensemble (multiple perspectives)",
                Strategy::Chain => "Chain (sequential reasoning)",
                Strategy::Parallel => "Parallel (merged approaches)",
            };

            info!(strategy = strategy_desc, "Strategy determined");
            if verbose {
                println!("   Strategy: {}", strategy_desc.yellow());
                println!();
            }
            strategy
        }
        .instrument(strategy_span)
        .await;

        // =========================================================================
        // Step 4: Select prompts based on strategy
        // =========================================================================

        let select_span = info_span!("select_prompts");
        let selected = async {
            info!("Selecting prompts based on strategy");
            if verbose {
                println!("{}", "Step 4: Selecting prompts...".cyan().bold());
            }

            let selected = orchestrator
                .select_prompts(&search_results, strategy)
                .await
                .unwrap_or_else(|_| Vec::new());

            info!(count = selected.len(), "Prompts selected");
            if verbose {
                println!(
                    "   Selected {} prompts:",
                    selected.len().to_string().green()
                );
                for prompt in &selected {
                    let role_badge = if prompt.role == "primary" {
                        "[Primary]".green()
                    } else if prompt.role.starts_with("step_") {
                        format!("[{}]", prompt.role).blue()
                    } else if prompt.role.starts_with("perspective_") {
                        format!("[{}]", prompt.role).magenta()
                    } else {
                        "[Supporting]".cyan()
                    };
                    println!("   - {} {}", role_badge, prompt.result.title);
                }
                println!();
            }
            selected
        }
        .instrument(select_span)
        .await;

        // =========================================================================
        // Step 5: Build meta-prompt
        // =========================================================================

        let build_span = info_span!("build_meta_prompt");
        let (meta_prompt, _meta_prompt_tokens) = async {
            info!("Building meta-prompt");
            if verbose {
                println!("{}", "Step 5: Building meta-prompt...".cyan().bold());
            }

            let meta_prompt = build_meta_prompt(&selected, &question);
            let meta_prompt_tokens = estimate_tokens(&meta_prompt);

            info!(tokens = meta_prompt_tokens, "Meta-prompt built");
            if verbose {
                println!(
                    "   Meta-prompt built ({} estimated tokens)",
                    meta_prompt_tokens.to_string().yellow()
                );
                println!();
            }
            (meta_prompt, meta_prompt_tokens)
        }
        .instrument(build_span)
        .await;

        // =========================================================================
        // Step 6: Call LLM
        // =========================================================================

        let llm_span = info_span!("llm_call", model = %config.llm_model);
        let (output_text, tokens_used, llm_duration) = async {
            info!("Calling LLM");
            if verbose {
                println!("{}", "Step 6: Calling LLM...".cyan().bold());
                println!("   Model: {}", config.llm_model.yellow());
            }

            // Create spinner for non-verbose mode
            let spinner = if !verbose {
                let s = ProgressBar::new_spinner();
                s.set_style(
                    ProgressStyle::default_spinner()
                        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
                        .template("{spinner:.cyan} {msg}")
                        .unwrap(),
                );
                s.set_message("Thinking...");
                s.enable_steady_tick(Duration::from_millis(80));
                Some(s)
            } else {
                None
            };

            let llm_start = Instant::now();
            let synthesis_result = synthesize_with_gemini(&gemini, &meta_prompt, &selected).await;
            let llm_duration = llm_start.elapsed().as_millis() as u64;

            // Stop spinner
            if let Some(s) = spinner {
                s.finish_and_clear();
            }

            let (output_text, tokens_used) = match synthesis_result {
                Ok(result) => {
                    info!(
                        duration_ms = llm_duration,
                        input_tokens = result.tokens.input,
                        output_tokens = result.tokens.output,
                        "LLM response received"
                    );
                    if verbose {
                        println!("   Response received in {}ms", llm_duration);
                        println!();
                    }
                    (result.output, result.tokens)
                }
                Err(e) => {
                    warn!(error = %e, duration_ms = llm_duration, "LLM call failed, using fallback");
                    if verbose {
                        println!("   {} LLM call failed: {}", "".red(), e);
                        println!("   Using meta-prompt as fallback output...");
                        println!();
                    }
                    // Fallback to showing the meta-prompt structure
                    let fallback = format!(
                        "LLM API error: {}\n\n--- Meta-Prompt (for debugging) ---\n\n{}",
                        e, meta_prompt
                    );
                    (
                        fallback.clone(),
                        TokenUsage {
                            input: estimate_tokens(&meta_prompt),
                            output: estimate_tokens(&fallback),
                            total: estimate_tokens(&meta_prompt) + estimate_tokens(&fallback),
                        },
                    )
                }
            };
            (output_text, tokens_used, llm_duration)
        }
        .instrument(llm_span)
        .await;

        // =========================================================================
        // Step 7: Validate response
        // =========================================================================

        let validate_span = info_span!("validate_response");
        let (spinoza_score, is_valid) = async {
            info!("Validating response with Spinoza principles");
            if verbose {
                println!("{}", "Step 7: Validating response...".cyan().bold());
            }

            let validation = validator.validate(&output_text).await;
            let (spinoza_score, is_valid) = match validation {
                Ok(v) => (v.scores.average(), v.is_valid),
                Err(_) => (0.5, true), // Default to acceptable if validation fails
            };

            info!(score = spinoza_score, is_valid = is_valid, "Validation complete");

            if verbose {
                let score_color = if spinoza_score >= 0.7 {
                    format!("{:.2}", spinoza_score).green()
                } else if spinoza_score >= 0.5 {
                    format!("{:.2}", spinoza_score).yellow()
                } else {
                    format!("{:.2}", spinoza_score).red()
                };

                let check = if is_valid { "✓" } else { "✗" };
                println!("   Spinoza score: {} {}", score_color, check);
                println!();
            }
            (spinoza_score, is_valid)
        }
        .instrument(validate_span)
        .await;

        // =========================================================================
        // Step 8: Apply corrections if needed
        // =========================================================================

        let correct_span = info_span!("apply_corrections");
        let final_output = async {
            if spinoza_score < 0.7 && !is_valid {
                info!("Applying corrections due to low validation score");
                if verbose {
                    println!("{}", "Step 8: Applying corrections...".cyan().bold());
                }

                // Detect ambiguities
                let ambiguities = corrector.detect_ambiguities(&output_text).await;

                match ambiguities {
                    Ok(ambs) if !ambs.is_empty() => {
                        debug!(count = ambs.len(), "Ambiguities detected");
                        if verbose {
                            println!("   Found {} ambiguities to address", ambs.len());
                        }

                        // Generate automatic answers for detected ambiguities
                        let answers: Vec<Answer> = ambs
                            .iter()
                            .enumerate()
                            .take(3) // Limit corrections
                            .map(|(i, amb)| {
                                Answer::new(i, format!("the specific {} in context", amb.text))
                            })
                            .collect();

                        if !answers.is_empty() {
                            match corrector.apply_corrections(&output_text, &answers).await {
                                Ok(result) => {
                                    info!(corrections = result.corrections_applied, "Corrections applied");
                                    if verbose {
                                        println!(
                                            "   Applied {} corrections",
                                            result.corrections_applied
                                        );
                                        println!();
                                    }
                                    result.content
                                }
                                Err(_) => output_text.clone(),
                            }
                        } else {
                            output_text.clone()
                        }
                    }
                    _ => {
                        debug!("No corrections needed");
                        if verbose {
                            println!("   No corrections needed");
                            println!();
                        }
                        output_text.clone()
                    }
                }
            } else {
                debug!("Skipping corrections - score acceptable");
                if verbose {
                    println!(
                        "{}",
                        "Step 8: Skipping corrections (score acceptable)".dimmed()
                    );
                    println!();
                }
                output_text.clone()
            }
        }
        .instrument(correct_span)
        .await;

        // =========================================================================
        // Final Output
        // =========================================================================

        let total_duration = pipeline_start.elapsed().as_millis() as u64;

        info!(
            total_duration_ms = total_duration,
            llm_duration_ms = llm_duration,
            prompts_used = selected.len(),
            input_tokens = tokens_used.input,
            output_tokens = tokens_used.output,
            spinoza_score = spinoza_score,
            "Ask pipeline complete"
        );

        // Print separator and final response
        println!("{}", "=".repeat(60).dimmed());
        println!();
        println!("{}", final_output);
        println!();

        // Print stats in verbose mode
        if verbose {
            println!("{}", "--- Stats ---".dimmed());
            println!("Prompts used: {}", selected.len());
            println!(
                "Tokens: {} in / {} out",
                tokens_used.input, tokens_used.output
            );
            println!("LLM duration: {}ms", llm_duration);
            println!("Total duration: {}ms", total_duration);
        }

        Ok(())
    }
    .instrument(span)
    .await
}

/// Build a meta-prompt from selected prompts and user question.
fn build_meta_prompt(
    prompts: &[panpsychism::orchestrator::SelectedPrompt],
    question: &str,
) -> String {
    let mut meta = String::new();

    meta.push_str("You are an expert assistant with access to curated knowledge. ");
    meta.push_str("Answer the user's question using insights from the following prompts.\n\n");

    meta.push_str("## Spinoza Principles\n\n");
    meta.push_str("Apply these philosophical principles in your response:\n");
    meta.push_str("- CONATUS: Strive to persist and enhance understanding\n");
    meta.push_str("- RATIO: Apply reason and logical analysis\n");
    meta.push_str("- LAETITIA: Generate responses that increase joy and clarity\n");
    meta.push_str("- NATURA: Align with the natural order of knowledge\n\n");

    if !prompts.is_empty() {
        meta.push_str("## Reference Prompts\n\n");

        for (i, prompt) in prompts.iter().enumerate() {
            meta.push_str(&format!(
                "### {} ({}) - Score: {:.2}\n",
                prompt.result.title, prompt.role, prompt.result.score
            ));
            meta.push_str(&prompt.result.excerpt);
            meta.push_str("\n\n");

            if i >= 4 {
                meta.push_str("... (additional context available)\n\n");
                break;
            }
        }
    }

    meta.push_str("## User Question\n\n");
    meta.push_str(question);
    meta.push_str("\n\n");

    meta.push_str("## Instructions\n\n");
    meta.push_str("1. Consider all provided prompts and their relevance to the question\n");
    meta.push_str("2. Synthesize a comprehensive, accurate response\n");
    meta.push_str("3. Be clear, logical, and helpful\n");
    meta.push_str(
        "4. If the prompts don't fully address the question, supplement with your knowledge\n",
    );
    meta.push_str("5. Aim for clarity and actionable insights\n");

    meta
}

/// Estimate token count for a string (rough approximation: ~4 chars per token).
fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

/// Local synthesis result struct.
struct SynthesisResultLocal {
    output: String,
    tokens: panpsychism::synthesizer::TokenUsage,
}

/// Synthesize a response using the Gemini client.
#[instrument(skip_all, fields(meta_prompt_len = meta_prompt.len(), prompts_count = _prompts.len()))]
async fn synthesize_with_gemini(
    gemini: &GeminiClient,
    meta_prompt: &str,
    _prompts: &[panpsychism::orchestrator::SelectedPrompt],
) -> anyhow::Result<SynthesisResultLocal> {
    use panpsychism::gemini::Message;
    use panpsychism::synthesizer::TokenUsage;

    debug!("Sending request to Gemini API");

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are a knowledgeable assistant that synthesizes information from curated prompts to provide accurate, helpful responses.".to_string(),
        },
        Message {
            role: "user".to_string(),
            content: meta_prompt.to_string(),
        },
    ];

    let response = gemini
        .chat(messages)
        .await
        .map_err(|e| anyhow::anyhow!("Gemini API call failed: {}", e))?;

    let output = response
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow::anyhow!("No response from LLM"))?;

    let tokens = response
        .usage
        .map(|u| TokenUsage {
            input: u.prompt_tokens,
            output: u.completion_tokens,
            total: u.total_tokens,
        })
        .unwrap_or_else(|| TokenUsage {
            input: estimate_tokens(meta_prompt),
            output: estimate_tokens(&output),
            total: estimate_tokens(meta_prompt) + estimate_tokens(&output),
        });

    debug!(
        input_tokens = tokens.input,
        output_tokens = tokens.output,
        "Gemini response received"
    );

    Ok(SynthesisResultLocal { output, tokens })
}

/// Shell usage statistics
#[derive(Debug, Default)]
struct ShellStats {
    /// Total questions asked in this session
    questions_asked: AtomicUsize,
    /// Total tokens used (input + output)
    total_tokens: AtomicUsize,
    /// Total time spent on LLM calls (milliseconds)
    total_llm_time_ms: AtomicU64,
    /// Session start time
    session_start: Option<Instant>,
}

impl ShellStats {
    fn new() -> Self {
        Self {
            questions_asked: AtomicUsize::new(0),
            total_tokens: AtomicUsize::new(0),
            total_llm_time_ms: AtomicU64::new(0),
            session_start: Some(Instant::now()),
        }
    }

    fn record_question(&self, tokens: usize, duration_ms: u64) {
        self.questions_asked.fetch_add(1, Ordering::Relaxed);
        self.total_tokens.fetch_add(tokens, Ordering::Relaxed);
        self.total_llm_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    fn display(&self) {
        let questions = self.questions_asked.load(Ordering::Relaxed);
        let tokens = self.total_tokens.load(Ordering::Relaxed);
        let llm_time = self.total_llm_time_ms.load(Ordering::Relaxed);
        let session_duration = self
            .session_start
            .map(|s| s.elapsed().as_secs())
            .unwrap_or(0);

        println!();
        println!("{}", "=== Session Statistics ===".cyan().bold());
        println!("  Questions asked: {}", questions.to_string().yellow());
        println!("  Total tokens:    {}", tokens.to_string().yellow());
        println!("  LLM time:        {:.2}s", llm_time as f64 / 1000.0);
        println!(
            "  Session time:    {}m {}s",
            session_duration / 60,
            session_duration % 60
        );
        if questions > 0 {
            println!("  Avg tokens/q:    {:.1}", tokens as f64 / questions as f64);
        }
        println!();
    }
}

/// Interactive shell mode with REPL capabilities.
#[instrument(skip_all)]
async fn cmd_shell() -> anyhow::Result<()> {
    let session_id = Uuid::new_v4();
    let span = info_span!("shell_session", id = %session_id);

    async move {
        info!("Starting interactive shell session");

        // Print banner
        println!();
        println!(
            "{}",
            r#"
  ____                              _     _
 |  _ \ __ _ _ __  _ __  ___ _   _  ___| |__ (_)___ _ __ ___
 | |_) / _` | '_ \| '_ \/ __| | | |/ __| '_ \| / __| '_ ` _ \
 |  __/ (_| | | | | |_) \__ \ |_| | (__| | | | \__ \ | | | | |
 |_|   \__,_|_| |_| .__/|___/\__, |\___|_| |_|_|___/_| |_| |_|
                  |_|        |___/
"#
            .cyan()
        );
        println!(
            "{}",
            format!("  Interactive Shell v{}", env!("CARGO_PKG_VERSION"))
                .cyan()
                .bold()
        );
        println!(
            "  {}",
            "Type /help for commands, or ask a question.".dimmed()
        );
        println!();

        // Load configuration
        let config = Config::load()?;
        debug!(model = %config.llm_model, "Shell configuration loaded");

        // Initialize readline editor with history
        let mut rl = DefaultEditor::new()?;

        // Get history path
        let history_path = dirs::home_dir()
            .map(|p| p.join(".panpsychism_history"))
            .unwrap_or_else(|| PathBuf::from(".panpsychism_history"));

        // Load history (ignore errors if file doesn't exist)
        let _ = rl.load_history(&history_path);

        // Initialize state
        let mut verbose = false;
        let stats = Arc::new(ShellStats::new());

        // Parse model from config string or use default
        let model = match config.llm_model.as_str() {
            "gemini-2.0-flash" => GeminiModel::Flash,
            "gemini-pro" => GeminiModel::Pro,
            "gemini-2.0-flash-thinking" => GeminiModel::ProThinking,
            "gemini-3-flash" => GeminiModel::Flash3,
            "gemini-3-pro-high" => GeminiModel::Pro3High,
            _ => GeminiModel::default(),
        };

        // Create Gemini client
        let client = GeminiClient::new()
            .with_endpoint(&config.llm_endpoint)
            .with_api_key(&config.llm_api_key)
            .with_model(model);

        loop {
            // Build prompt with mode indicator
            let prompt = if verbose {
                format!("{} ", "D >".yellow())
            } else {
                format!("{} ", "> ".cyan())
            };

            match rl.readline(&prompt) {
                Ok(line) => {
                    let trimmed = line.trim();

                    // Skip empty lines
                    if trimmed.is_empty() {
                        continue;
                    }

                    // Add to history
                    let _ = rl.add_history_entry(&line);

                    // Process built-in commands
                    match trimmed {
                        "/help" | "/h" | "/?" => {
                            debug!("Help command invoked");
                            shell_print_help();
                        }
                        "/clear" | "/cls" => {
                            debug!("Clear screen command invoked");
                            shell_clear_screen();
                        }
                        "/exit" | "/quit" | "/q" => {
                            info!("Exit command invoked");
                            break;
                        }
                        "/config" | "/cfg" => {
                            debug!("Config display command invoked");
                            shell_show_config(&config);
                        }
                        "/verbose" | "/v" => {
                            verbose = !verbose;
                            debug!(verbose = verbose, "Verbose mode toggled");
                            println!(
                                "Verbose mode: {}",
                                if verbose {
                                    "ON".green().bold()
                                } else {
                                    "OFF".red().bold()
                                }
                            );
                        }
                        "/stats" | "/s" => {
                            debug!("Stats display command invoked");
                            stats.display();
                        }
                        "/version" => {
                            println!(
                                "Panpsychism v{} - {}",
                                env!("CARGO_PKG_VERSION"),
                                env!("CARGO_PKG_DESCRIPTION")
                            );
                        }
                        cmd if cmd.starts_with('/') => {
                            warn!(command = cmd, "Unknown command");
                            println!(
                                "{} Unknown command: {}",
                                "Error:".red().bold(),
                                cmd.yellow()
                            );
                            println!("Type {} for available commands.", "/help".cyan());
                        }
                        question => {
                            // Process as a question
                            if let Err(e) =
                                shell_process_question(&client, question, verbose, &stats).await
                            {
                                error!(error = %e, "Question processing failed");
                                println!("{} {}", "Error:".red().bold(), e);
                            }
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    // Ctrl+C - show message but don't exit
                    debug!("Interrupted (Ctrl+C)");
                    println!("^C (type /exit to quit)");
                }
                Err(ReadlineError::Eof) => {
                    // Ctrl+D - exit
                    info!("EOF received, exiting");
                    break;
                }
                Err(err) => {
                    error!(error = ?err, "Readline error");
                    println!("{} {:?}", "Readline error:".red().bold(), err);
                    break;
                }
            }
        }

        // Save history
        if let Err(e) = rl.save_history(&history_path) {
            warn!(error = %e, "Failed to save history");
            eprintln!(
                "{} Failed to save history: {}",
                "Warning:".yellow().bold(),
                e
            );
        }

        // Show final stats
        stats.display();
        info!("Shell session ended");
        println!("{}", "Goodbye!".cyan().bold());

        Ok(())
    }
    .instrument(span)
    .await
}

/// Process a question through the LLM pipeline.
#[instrument(skip(client, stats), fields(question_len = question.len()))]
async fn shell_process_question(
    client: &GeminiClient,
    question: &str,
    verbose: bool,
    stats: &Arc<ShellStats>,
) -> anyhow::Result<()> {
    info!(question = %question, "Processing shell question");

    if verbose {
        println!("{}", "--- Processing ---".dimmed());
        println!("Query: {}", question.yellow());
    }

    // Create and start spinner
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
            .template("{spinner:.cyan} {msg}")?,
    );
    spinner.set_message("Thinking...");
    spinner.enable_steady_tick(Duration::from_millis(80));

    let start = Instant::now();

    // Call LLM
    let result = client.complete(question).await;

    let duration = start.elapsed();
    spinner.finish_and_clear();

    match result {
        Ok(response) => {
            // Estimate tokens (rough estimate: ~4 chars per token)
            let estimated_tokens = (question.len() + response.len()) / 4;

            // Record stats
            stats.record_question(estimated_tokens, duration.as_millis() as u64);

            info!(
                duration_ms = duration.as_millis() as u64,
                estimated_tokens = estimated_tokens,
                "Shell question answered"
            );

            if verbose {
                println!(
                    "{} {:.2}s | ~{} tokens",
                    "Response:".green().bold(),
                    duration.as_secs_f64(),
                    estimated_tokens
                );
                println!();
            }

            // Print response with nice formatting
            println!();
            for line in response.lines() {
                println!("{}", line);
            }
            println!();
        }
        Err(e) => {
            error!(error = %e, "Shell LLM call failed");
            return Err(anyhow::anyhow!("LLM call failed: {}", e));
        }
    }

    Ok(())
}

/// Print help message for shell commands.
fn shell_print_help() {
    println!();
    println!("{}", "=== Panpsychism Shell Commands ===".cyan().bold());
    println!();
    println!("  {}    Show this help message", "/help, /h, /?".green());
    println!("  {}     Clear the screen", "/clear, /cls".green());
    println!("  {}  Exit the shell", "/exit, /quit, /q".green());
    println!("  {}   Show current configuration", "/config, /cfg".green());
    println!("  {}   Toggle verbose mode", "/verbose, /v".green());
    println!("  {}    Show session statistics", "/stats, /s".green());
    println!("  {}      Show version information", "/version".green());
    println!();
    println!(
        "  {}",
        "Any other input is treated as a question for the AI.".dimmed()
    );
    println!();
    println!("{}", "Keyboard Shortcuts:".cyan().bold());
    println!("  {}          Show interrupt message", "Ctrl+C".green());
    println!("  {}          Exit the shell", "Ctrl+D".green());
    println!("  {}  Navigate command history", "Up/Down".green());
    println!();
}

/// Clear the terminal screen.
fn shell_clear_screen() {
    // ANSI escape sequence to clear screen and move cursor to top-left
    print!("\x1B[2J\x1B[1;1H");
    // Flush stdout to ensure immediate effect
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

/// Show current configuration.
fn shell_show_config(config: &Config) {
    println!();
    println!("{}", "=== Current Configuration ===".cyan().bold());
    println!();
    println!(
        "  {} {}",
        "Prompts directory:".green(),
        config.prompts_dir.display()
    );
    println!(
        "  {} {}",
        "Data directory:   ".green(),
        config.data_dir.display()
    );
    println!(
        "  {} {}",
        "Index file:       ".green(),
        config.index_file.display()
    );
    println!("  {} {}", "LLM endpoint:     ".green(), config.llm_endpoint);
    println!("  {} {}", "LLM model:        ".green(), config.llm_model);
    println!(
        "  {} {:?}",
        "Privacy tier:     ".green(),
        config.privacy.tier
    );
    println!();

    // Show config file location
    let config_path = Config::config_path();
    if config_path.exists() {
        println!(
            "  {} {}",
            "Config file:".dimmed(),
            config_path.display().to_string().dimmed()
        );
    } else {
        println!(
            "  {} {}",
            "Config file:".dimmed(),
            "(using defaults)".dimmed()
        );
    }
    println!();
}
