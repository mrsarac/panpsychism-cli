//! Documenter Agent module for Project Panpsychism.
//!
//! The Knowledge Recorder - generates and maintains documentation in various formats.
//! Like a scholarly scribe in an ancient library, this agent transforms knowledge
//! into well-structured, accessible documentation.
//!
//! # The Scholar's Quill Metaphor
//!
//! In the realm of Project Panpsychism, the Documenter Agent acts as the
//! **Knowledge Recorder** - a meticulous scholar who transcribes wisdom
//! into scrolls that can be read by all. Just as a master scribe organizes
//! knowledge into illuminated manuscripts, this agent structures content
//! into clear, navigable documentation.
//!
//! ## Documentation Formats
//!
//! The Documenter supports multiple output formats:
//!
//! - **Markdown**: Universal, readable, version-control friendly
//! - **HTML**: Rich formatting for web presentation
//! - **PDF**: Polished documents for distribution
//! - **AsciiDoc**: Technical documentation with advanced features
//! - **reStructuredText**: Python ecosystem documentation standard
//!
//! # Philosophical Foundation
//!
//! Following Spinoza's principles:
//!
//! - **CONATUS**: Preserve the self-sustaining clarity of knowledge
//! - **RATIO**: Apply logical organization and structure
//! - **LAETITIA**: Enhance understanding through clear presentation
//! - **NATURA**: Respect the natural flow of information
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::documenter::{DocumenterAgent, DocFormat, Audience};
//! use panpsychism::gemini::GeminiClient;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let client = Arc::new(GeminiClient::new());
//!     let documenter = DocumenterAgent::builder(client)
//!         .with_format(DocFormat::Markdown)
//!         .with_audience(Audience::Developer)
//!         .build();
//!
//!     let content = "Explain how the authentication system works...";
//!     let doc = documenter.document(content).await?;
//!
//!     println!("Generated documentation:\n{}", doc.render());
//!     Ok(())
//! }
//! ```

use crate::gemini::{GeminiClient, Message};
use crate::{Error, Result};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use std::time::Instant;

// =============================================================================
// DOCUMENTATION FORMAT
// =============================================================================

/// Output format for generated documentation.
///
/// Each format has its own rendering characteristics and use cases:
///
/// - **Markdown**: Universal, works everywhere, version-control friendly
/// - **Html**: Rich formatting, embedded styling, web-ready
/// - **Pdf**: Print-ready, polished appearance (requires post-processing)
/// - **Asciidoc**: Technical docs with advanced features
/// - **RestructuredText**: Python/Sphinx ecosystem standard
///
/// # The Scribe's Ink Choice
///
/// Just as a scribe chooses different inks and parchments for different scrolls,
/// the documentation format determines how knowledge is presented and consumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DocFormat {
    /// Markdown format - universal and readable.
    #[default]
    Markdown,

    /// HTML format - rich formatting for web.
    Html,

    /// PDF-ready format - for polished distribution.
    Pdf,

    /// AsciiDoc format - advanced technical documentation.
    Asciidoc,

    /// reStructuredText format - Python ecosystem standard.
    RestructuredText,
}

impl DocFormat {
    /// Get the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            DocFormat::Markdown => "md",
            DocFormat::Html => "html",
            DocFormat::Pdf => "pdf",
            DocFormat::Asciidoc => "adoc",
            DocFormat::RestructuredText => "rst",
        }
    }

    /// Get the MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            DocFormat::Markdown => "text/markdown",
            DocFormat::Html => "text/html",
            DocFormat::Pdf => "application/pdf",
            DocFormat::Asciidoc => "text/asciidoc",
            DocFormat::RestructuredText => "text/x-rst",
        }
    }

    /// Get a human-readable label for this format.
    pub fn label(&self) -> &'static str {
        match self {
            DocFormat::Markdown => "Markdown",
            DocFormat::Html => "HTML",
            DocFormat::Pdf => "PDF",
            DocFormat::Asciidoc => "AsciiDoc",
            DocFormat::RestructuredText => "reStructuredText",
        }
    }

    /// Check if this format supports rich styling.
    pub fn supports_styling(&self) -> bool {
        matches!(self, DocFormat::Html | DocFormat::Pdf | DocFormat::Asciidoc)
    }

    /// Check if this format is plain text based.
    pub fn is_plain_text(&self) -> bool {
        matches!(
            self,
            DocFormat::Markdown | DocFormat::Asciidoc | DocFormat::RestructuredText
        )
    }
}

impl std::fmt::Display for DocFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

impl std::str::FromStr for DocFormat {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "markdown" | "md" => Ok(DocFormat::Markdown),
            "html" | "htm" => Ok(DocFormat::Html),
            "pdf" => Ok(DocFormat::Pdf),
            "asciidoc" | "adoc" => Ok(DocFormat::Asciidoc),
            "restructuredtext" | "rst" => Ok(DocFormat::RestructuredText),
            _ => Err(Error::Config(format!(
                "Unknown documentation format: '{}'. Valid formats: markdown, html, pdf, asciidoc, rst",
                s
            ))),
        }
    }
}

// =============================================================================
// TARGET AUDIENCE
// =============================================================================

/// Target audience for the documentation.
///
/// The audience determines the level of detail, terminology, and
/// assumptions made in the documentation.
///
/// # The Reader's Perspective
///
/// Like a skilled teacher adapting their explanation to their students,
/// the documentation adapts its language and depth to its intended readers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Audience {
    /// Software developers - technical, assumes coding knowledge.
    #[default]
    Developer,

    /// End users - non-technical, focuses on usage.
    EndUser,

    /// System administrators - operational focus.
    Administrator,

    /// Beginners - extra explanations, no assumptions.
    Beginner,

    /// Experts - concise, advanced concepts.
    Expert,
}

impl Audience {
    /// Get a description of this audience type.
    pub fn description(&self) -> &'static str {
        match self {
            Audience::Developer => "Software developers with coding experience",
            Audience::EndUser => "Non-technical end users of the product",
            Audience::Administrator => "System administrators and DevOps engineers",
            Audience::Beginner => "Newcomers with little to no prior knowledge",
            Audience::Expert => "Advanced users seeking in-depth information",
        }
    }

    /// Get the expected technical level (0-4).
    pub fn technical_level(&self) -> u8 {
        match self {
            Audience::Beginner => 0,
            Audience::EndUser => 1,
            Audience::Administrator => 2,
            Audience::Developer => 3,
            Audience::Expert => 4,
        }
    }

    /// Get a label for this audience.
    pub fn label(&self) -> &'static str {
        match self {
            Audience::Developer => "Developer",
            Audience::EndUser => "End User",
            Audience::Administrator => "Administrator",
            Audience::Beginner => "Beginner",
            Audience::Expert => "Expert",
        }
    }

    /// Check if this audience expects code examples.
    pub fn expects_code_examples(&self) -> bool {
        matches!(self, Audience::Developer | Audience::Expert)
    }

    /// Check if this audience needs detailed explanations.
    pub fn needs_detailed_explanations(&self) -> bool {
        matches!(self, Audience::Beginner | Audience::EndUser)
    }
}

impl std::fmt::Display for Audience {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

impl std::str::FromStr for Audience {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "developer" | "dev" => Ok(Audience::Developer),
            "enduser" | "end-user" | "user" => Ok(Audience::EndUser),
            "administrator" | "admin" | "ops" => Ok(Audience::Administrator),
            "beginner" | "newbie" | "novice" => Ok(Audience::Beginner),
            "expert" | "advanced" => Ok(Audience::Expert),
            _ => Err(Error::Config(format!(
                "Unknown audience: '{}'. Valid: developer, enduser, administrator, beginner, expert",
                s
            ))),
        }
    }
}

// =============================================================================
// DOCUMENTATION SECTION
// =============================================================================

/// A section within the documentation.
///
/// Represents a logical unit of documentation with a heading,
/// content, and optional examples and cross-references.
///
/// # The Chapter Metaphor
///
/// Like chapters in a book, sections organize knowledge into
/// digestible units that can be navigated and referenced.
#[derive(Debug, Clone)]
pub struct DocSection {
    /// Section heading/title.
    pub heading: String,

    /// Section content (body text).
    pub content: String,

    /// Heading level (1 = top-level, 2 = subsection, etc.).
    pub level: u8,

    /// Code or usage examples for this section.
    pub examples: Vec<String>,

    /// Cross-references to other sections or external resources.
    pub cross_refs: Vec<String>,
}

impl DocSection {
    /// Create a new documentation section.
    ///
    /// # Arguments
    ///
    /// * `heading` - The section title
    /// * `content` - The section body text
    pub fn new(heading: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            heading: heading.into(),
            content: content.into(),
            level: 1,
            examples: Vec::new(),
            cross_refs: Vec::new(),
        }
    }

    /// Set the heading level.
    pub fn with_level(mut self, level: u8) -> Self {
        self.level = level.clamp(1, 6);
        self
    }

    /// Add an example to this section.
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.examples.push(example.into());
        self
    }

    /// Add multiple examples to this section.
    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples.extend(examples);
        self
    }

    /// Add a cross-reference to this section.
    pub fn with_cross_ref(mut self, reference: impl Into<String>) -> Self {
        self.cross_refs.push(reference.into());
        self
    }

    /// Add multiple cross-references.
    pub fn with_cross_refs(mut self, refs: Vec<String>) -> Self {
        self.cross_refs.extend(refs);
        self
    }

    /// Check if this section has examples.
    pub fn has_examples(&self) -> bool {
        !self.examples.is_empty()
    }

    /// Check if this section has cross-references.
    pub fn has_cross_refs(&self) -> bool {
        !self.cross_refs.is_empty()
    }

    /// Get the word count of the content.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    /// Render the section to Markdown.
    pub fn render_markdown(&self) -> String {
        let heading_prefix = "#".repeat(self.level as usize);
        let mut output = format!("{} {}\n\n{}\n", heading_prefix, self.heading, self.content);

        if !self.examples.is_empty() {
            output.push_str("\n**Examples:**\n\n");
            for example in &self.examples {
                output.push_str(&format!("```\n{}\n```\n\n", example));
            }
        }

        if !self.cross_refs.is_empty() {
            output.push_str("\n**See also:**\n\n");
            for reference in &self.cross_refs {
                output.push_str(&format!("- {}\n", reference));
            }
        }

        output
    }

    /// Render the section to HTML.
    pub fn render_html(&self) -> String {
        let tag = format!("h{}", self.level.min(6));
        let mut output = format!(
            "<{}>{}</{}>\n<p>{}</p>\n",
            tag,
            html_escape(&self.heading),
            tag,
            html_escape(&self.content)
        );

        if !self.examples.is_empty() {
            output.push_str("<h4>Examples</h4>\n");
            for example in &self.examples {
                output.push_str(&format!("<pre><code>{}</code></pre>\n", html_escape(example)));
            }
        }

        if !self.cross_refs.is_empty() {
            output.push_str("<h4>See also</h4>\n<ul>\n");
            for reference in &self.cross_refs {
                output.push_str(&format!("<li>{}</li>\n", html_escape(reference)));
            }
            output.push_str("</ul>\n");
        }

        output
    }

    /// Render the section to AsciiDoc.
    pub fn render_asciidoc(&self) -> String {
        let heading_prefix = "=".repeat(self.level as usize + 1);
        let mut output = format!("{} {}\n\n{}\n", heading_prefix, self.heading, self.content);

        if !self.examples.is_empty() {
            output.push_str("\n.Examples\n");
            for example in &self.examples {
                output.push_str(&format!("[source]\n----\n{}\n----\n\n", example));
            }
        }

        if !self.cross_refs.is_empty() {
            output.push_str("\n.See also\n");
            for reference in &self.cross_refs {
                output.push_str(&format!("* {}\n", reference));
            }
        }

        output
    }

    /// Render the section to reStructuredText.
    pub fn render_rst(&self) -> String {
        let underline_char = match self.level {
            1 => '=',
            2 => '-',
            3 => '~',
            4 => '^',
            _ => '"',
        };
        let underline = underline_char.to_string().repeat(self.heading.len());

        let mut output = format!("{}\n{}\n\n{}\n", self.heading, underline, self.content);

        if !self.examples.is_empty() {
            output.push_str("\n**Examples:**\n\n");
            for example in &self.examples {
                output.push_str(&format!("::\n\n    {}\n\n", example.replace('\n', "\n    ")));
            }
        }

        if !self.cross_refs.is_empty() {
            output.push_str("\n**See also:**\n\n");
            for reference in &self.cross_refs {
                output.push_str(&format!("* {}\n", reference));
            }
        }

        output
    }
}

impl std::fmt::Display for DocSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render_markdown())
    }
}

// =============================================================================
// DOCUMENTATION
// =============================================================================

/// Complete generated documentation.
///
/// Contains all sections along with metadata about the documentation
/// generation process.
///
/// # The Complete Manuscript
///
/// Like a finished manuscript with its title, chapters, and
/// publication details, this structure holds the complete
/// documentation ready for distribution.
#[derive(Debug, Clone)]
pub struct Documentation {
    /// Document title.
    pub title: String,

    /// Document sections.
    pub sections: Vec<DocSection>,

    /// Output format.
    pub format: DocFormat,

    /// Generation timestamp.
    pub generated_at: DateTime<Utc>,

    /// Document version.
    pub version: String,

    /// Target audience.
    pub audience: Audience,

    /// Processing time in milliseconds.
    pub duration_ms: u64,

    /// Token usage for generation.
    pub tokens_used: usize,
}

impl Documentation {
    /// Create a new documentation instance.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            sections: Vec::new(),
            format: DocFormat::default(),
            generated_at: Utc::now(),
            version: "1.0.0".to_string(),
            audience: Audience::default(),
            duration_ms: 0,
            tokens_used: 0,
        }
    }

    /// Add a section to the documentation.
    pub fn with_section(mut self, section: DocSection) -> Self {
        self.sections.push(section);
        self
    }

    /// Add multiple sections.
    pub fn with_sections(mut self, sections: Vec<DocSection>) -> Self {
        self.sections.extend(sections);
        self
    }

    /// Set the format.
    pub fn with_format(mut self, format: DocFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the audience.
    pub fn with_audience(mut self, audience: Audience) -> Self {
        self.audience = audience;
        self
    }

    /// Get the total word count.
    pub fn word_count(&self) -> usize {
        self.sections.iter().map(|s| s.word_count()).sum()
    }

    /// Get the total number of sections.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Get the total number of examples.
    pub fn example_count(&self) -> usize {
        self.sections.iter().map(|s| s.examples.len()).sum()
    }

    /// Generate a table of contents.
    pub fn table_of_contents(&self) -> Vec<(u8, String)> {
        self.sections
            .iter()
            .map(|s| (s.level, s.heading.clone()))
            .collect()
    }

    /// Render the documentation to the configured format.
    pub fn render(&self) -> String {
        match self.format {
            DocFormat::Markdown => self.render_markdown(),
            DocFormat::Html => self.render_html(),
            DocFormat::Pdf => self.render_pdf_source(),
            DocFormat::Asciidoc => self.render_asciidoc(),
            DocFormat::RestructuredText => self.render_rst(),
        }
    }

    /// Render to Markdown format.
    pub fn render_markdown(&self) -> String {
        let mut output = format!("# {}\n\n", self.title);

        // Metadata
        output.push_str(&format!(
            "> Generated: {} | Version: {} | Audience: {}\n\n",
            self.generated_at.format("%Y-%m-%d %H:%M:%S UTC"),
            self.version,
            self.audience
        ));

        // Table of contents
        if self.sections.len() > 1 {
            output.push_str("## Table of Contents\n\n");
            for (level, heading) in self.table_of_contents() {
                let indent = "  ".repeat((level - 1) as usize);
                let anchor = heading.to_lowercase().replace(' ', "-");
                output.push_str(&format!("{}- [{}](#{})\n", indent, heading, anchor));
            }
            output.push_str("\n---\n\n");
        }

        // Sections
        for section in &self.sections {
            output.push_str(&section.render_markdown());
            output.push('\n');
        }

        output
    }

    /// Render to HTML format.
    pub fn render_html(&self) -> String {
        let mut output = String::from(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>"#,
        );

        output.push_str(&html_escape(&self.title));
        output.push_str(
            r#"</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; line-height: 1.6; }
        pre { background: #f4f4f4; padding: 1rem; overflow-x: auto; border-radius: 4px; }
        code { font-family: 'Fira Code', Consolas, Monaco, monospace; }
        .metadata { color: #666; font-size: 0.9em; margin-bottom: 2rem; }
        .toc { background: #f9f9f9; padding: 1rem; border-radius: 4px; margin-bottom: 2rem; }
        .toc ul { margin: 0; }
    </style>
</head>
<body>
"#,
        );

        output.push_str(&format!("<h1>{}</h1>\n", html_escape(&self.title)));

        // Metadata
        output.push_str(&format!(
            "<p class=\"metadata\">Generated: {} | Version: {} | Audience: {}</p>\n",
            self.generated_at.format("%Y-%m-%d %H:%M:%S UTC"),
            self.version,
            self.audience
        ));

        // Table of contents
        if self.sections.len() > 1 {
            output.push_str("<nav class=\"toc\">\n<h2>Table of Contents</h2>\n<ul>\n");
            for (_, heading) in self.table_of_contents() {
                let anchor = heading.to_lowercase().replace(' ', "-");
                output.push_str(&format!(
                    "<li><a href=\"#{}\">{}</a></li>\n",
                    anchor,
                    html_escape(&heading)
                ));
            }
            output.push_str("</ul>\n</nav>\n");
        }

        // Sections
        for section in &self.sections {
            let anchor = section.heading.to_lowercase().replace(' ', "-");
            output.push_str(&format!("<section id=\"{}\">\n", anchor));
            output.push_str(&section.render_html());
            output.push_str("</section>\n");
        }

        output.push_str("</body>\n</html>\n");

        output
    }

    /// Render to PDF-ready source (LaTeX-style markdown).
    pub fn render_pdf_source(&self) -> String {
        // For PDF, we generate enhanced Markdown that can be processed by pandoc
        let mut output = format!("---\ntitle: \"{}\"\n", self.title);
        output.push_str(&format!("date: \"{}\"\n", self.generated_at.format("%Y-%m-%d")));
        output.push_str(&format!("version: \"{}\"\n", self.version));
        output.push_str("documentclass: article\n");
        output.push_str("---\n\n");

        output.push_str(&self.render_markdown());

        output
    }

    /// Render to AsciiDoc format.
    pub fn render_asciidoc(&self) -> String {
        let mut output = format!("= {}\n", self.title);
        output.push_str(&format!(
            ":docdate: {}\n",
            self.generated_at.format("%Y-%m-%d")
        ));
        output.push_str(&format!(":revnumber: {}\n", self.version));
        output.push_str(":toc:\n:toclevels: 3\n\n");

        for section in &self.sections {
            output.push_str(&section.render_asciidoc());
            output.push('\n');
        }

        output
    }

    /// Render to reStructuredText format.
    pub fn render_rst(&self) -> String {
        let title_underline = "=".repeat(self.title.len());
        let mut output = format!("{}\n{}\n\n", self.title, title_underline);

        output.push_str(&format!(
            "| **Generated:** {}\n",
            self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        output.push_str(&format!("| **Version:** {}\n", self.version));
        output.push_str(&format!("| **Audience:** {}\n\n", self.audience));

        // Table of contents
        output.push_str(".. contents::\n   :depth: 3\n\n");

        for section in &self.sections {
            output.push_str(&section.render_rst());
            output.push('\n');
        }

        output
    }

    /// Get statistics about the documentation.
    pub fn stats(&self) -> String {
        format!(
            "{} sections, {} words, {} examples, generated in {}ms",
            self.section_count(),
            self.word_count(),
            self.example_count(),
            self.duration_ms
        )
    }
}

impl std::fmt::Display for Documentation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render())
    }
}

// =============================================================================
// DOCUMENTER CONFIGURATION
// =============================================================================

/// Configuration for the documenter agent.
#[derive(Debug, Clone)]
pub struct DocConfig {
    /// Include code examples in documentation.
    pub include_examples: bool,

    /// Include API reference section.
    pub include_api_reference: bool,

    /// Maximum section depth (1-6).
    pub max_depth: u8,

    /// Target audience for the documentation.
    pub target_audience: Audience,

    /// Output format.
    pub format: DocFormat,

    /// Include table of contents.
    pub include_toc: bool,

    /// Include cross-references.
    pub include_cross_refs: bool,

    /// Maximum retries for LLM calls.
    pub max_retries: u32,

    /// Document version string.
    pub version: String,
}

impl Default for DocConfig {
    fn default() -> Self {
        Self {
            include_examples: true,
            include_api_reference: false,
            max_depth: 3,
            target_audience: Audience::Developer,
            format: DocFormat::Markdown,
            include_toc: true,
            include_cross_refs: true,
            max_retries: 3,
            version: "1.0.0".to_string(),
        }
    }
}

impl DocConfig {
    /// Create a configuration for API documentation.
    pub fn for_api() -> Self {
        Self {
            include_examples: true,
            include_api_reference: true,
            target_audience: Audience::Developer,
            ..Default::default()
        }
    }

    /// Create a configuration for user guides.
    pub fn for_users() -> Self {
        Self {
            include_examples: true,
            include_api_reference: false,
            target_audience: Audience::EndUser,
            ..Default::default()
        }
    }

    /// Create a configuration for quick reference.
    pub fn for_quick_reference() -> Self {
        Self {
            include_examples: true,
            include_api_reference: false,
            max_depth: 2,
            include_toc: false,
            target_audience: Audience::Expert,
            ..Default::default()
        }
    }
}

// =============================================================================
// DOCUMENTER AGENT BUILDER
// =============================================================================

/// Builder for DocumenterAgent configuration.
///
/// Provides a fluent interface for configuring the documenter agent
/// before construction.
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::documenter::{DocumenterAgentBuilder, DocFormat, Audience};
/// use std::sync::Arc;
///
/// let agent = DocumenterAgentBuilder::new(Arc::new(GeminiClient::new()))
///     .with_format(DocFormat::Html)
///     .with_audience(Audience::Developer)
///     .with_examples(true)
///     .with_max_depth(4)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct DocumenterAgentBuilder {
    client: Arc<GeminiClient>,
    config: DocConfig,
}

impl DocumenterAgentBuilder {
    /// Create a new builder with the given Gemini client.
    pub fn new(client: Arc<GeminiClient>) -> Self {
        Self {
            client,
            config: DocConfig::default(),
        }
    }

    /// Set the output format.
    pub fn with_format(mut self, format: DocFormat) -> Self {
        self.config.format = format;
        self
    }

    /// Set the target audience.
    pub fn with_audience(mut self, audience: Audience) -> Self {
        self.config.target_audience = audience;
        self
    }

    /// Set whether to include examples.
    pub fn with_examples(mut self, include: bool) -> Self {
        self.config.include_examples = include;
        self
    }

    /// Set whether to include API reference.
    pub fn with_api_reference(mut self, include: bool) -> Self {
        self.config.include_api_reference = include;
        self
    }

    /// Set the maximum section depth.
    pub fn with_max_depth(mut self, depth: u8) -> Self {
        self.config.max_depth = depth.clamp(1, 6);
        self
    }

    /// Set whether to include table of contents.
    pub fn with_toc(mut self, include: bool) -> Self {
        self.config.include_toc = include;
        self
    }

    /// Set whether to include cross-references.
    pub fn with_cross_refs(mut self, include: bool) -> Self {
        self.config.include_cross_refs = include;
        self
    }

    /// Set the maximum retry count.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Set the document version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.config.version = version.into();
        self
    }

    /// Set the entire configuration.
    pub fn with_config(mut self, config: DocConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the DocumenterAgent.
    pub fn build(self) -> DocumenterAgent {
        DocumenterAgent {
            client: self.client,
            config: self.config,
        }
    }
}

// =============================================================================
// DOCUMENTER AGENT
// =============================================================================

/// The Knowledge Recorder - Agent 34 of Project Panpsychism.
///
/// This agent generates and maintains documentation in various formats,
/// using the Scholar's Quill metaphor. It transforms knowledge into
/// well-structured, accessible documentation.
///
/// # Architecture
///
/// The DocumenterAgent uses an `Arc<GeminiClient>` for LLM integration,
/// allowing shared access across multiple operations. It supports:
///
/// - Multiple output formats (Markdown, HTML, PDF, AsciiDoc, RST)
/// - Audience-aware content generation
/// - Code example generation
/// - Cross-reference management
/// - Table of contents generation
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::documenter::{DocumenterAgent, DocFormat, Audience};
/// use panpsychism::gemini::GeminiClient;
/// use std::sync::Arc;
///
/// let client = Arc::new(GeminiClient::new());
/// let documenter = DocumenterAgent::builder(client)
///     .with_format(DocFormat::Markdown)
///     .with_audience(Audience::Developer)
///     .build();
///
/// // Generate documentation
/// let doc = documenter.document("Authentication system overview").await?;
/// println!("{}", doc.render());
///
/// // Generate API reference
/// let api_doc = documenter.generate_api_reference("User API endpoints").await?;
/// ```
#[derive(Debug, Clone)]
pub struct DocumenterAgent {
    /// The Gemini client for LLM calls.
    client: Arc<GeminiClient>,

    /// Configuration for documentation behavior.
    config: DocConfig,
}

impl DocumenterAgent {
    /// Create a new DocumenterAgent with the given Gemini client.
    ///
    /// # Arguments
    ///
    /// * `client` - Arc-wrapped GeminiClient for LLM integration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = Arc::new(GeminiClient::new());
    /// let documenter = DocumenterAgent::new(client);
    /// ```
    pub fn new(client: Arc<GeminiClient>) -> Self {
        Self {
            client,
            config: DocConfig::default(),
        }
    }

    /// Create a new DocumenterAgent with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `client` - Arc-wrapped GeminiClient for LLM integration
    /// * `config` - Custom configuration for documentation
    pub fn with_config(client: Arc<GeminiClient>, config: DocConfig) -> Self {
        Self { client, config }
    }

    /// Create a builder for fluent configuration.
    ///
    /// # Arguments
    ///
    /// * `client` - Arc-wrapped GeminiClient for LLM integration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let documenter = DocumenterAgent::builder(client)
    ///     .with_format(DocFormat::Html)
    ///     .with_audience(Audience::Developer)
    ///     .build();
    /// ```
    pub fn builder(client: Arc<GeminiClient>) -> DocumenterAgentBuilder {
        DocumenterAgentBuilder::new(client)
    }

    /// Get the current configuration.
    pub fn config(&self) -> &DocConfig {
        &self.config
    }

    // =========================================================================
    // CORE DOCUMENTATION
    // =========================================================================

    /// Generate documentation from content.
    ///
    /// This is the primary documentation method. It analyzes the input content
    /// and generates structured documentation appropriate for the configured
    /// audience and format.
    ///
    /// # The Scribe's Art
    ///
    /// Like a master scribe transforming raw knowledge into a polished manuscript,
    /// this method organizes content into clear, navigable sections.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to document
    ///
    /// # Returns
    ///
    /// A `Documentation` containing structured sections and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Content is empty
    /// - LLM call fails after retries
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let doc = documenter.document(
    ///     "Explain the authentication flow including OAuth2 and JWT tokens"
    /// ).await?;
    ///
    /// println!("Generated {} sections", doc.section_count());
    /// println!("{}", doc.render());
    /// ```
    pub async fn document(&self, content: &str) -> Result<Documentation> {
        let start = Instant::now();

        // Validate input
        if content.trim().is_empty() {
            return Err(Error::Synthesis(
                "Cannot generate documentation for empty content".to_string(),
            ));
        }

        let prompt = self.build_document_prompt(content);
        let response = self.call_llm(&prompt).await?;
        let sections = self.parse_sections(&response);

        let duration_ms = start.elapsed().as_millis() as u64;

        let doc = Documentation {
            title: self.extract_title(&response, content),
            sections,
            format: self.config.format,
            generated_at: Utc::now(),
            version: self.config.version.clone(),
            audience: self.config.target_audience,
            duration_ms,
            tokens_used: Self::estimate_tokens(content) + Self::estimate_tokens(&prompt),
        };

        Ok(doc)
    }

    /// Generate documentation with a specific title.
    ///
    /// # Arguments
    ///
    /// * `title` - The document title
    /// * `content` - The content to document
    ///
    /// # Returns
    ///
    /// A `Documentation` with the specified title.
    pub async fn document_with_title(
        &self,
        title: impl Into<String>,
        content: &str,
    ) -> Result<Documentation> {
        let mut doc = self.document(content).await?;
        doc.title = title.into();
        Ok(doc)
    }

    /// Generate API reference documentation.
    ///
    /// Creates documentation specifically structured for API reference,
    /// with emphasis on endpoints, parameters, and response formats.
    ///
    /// # Arguments
    ///
    /// * `api_description` - Description of the API to document
    ///
    /// # Returns
    ///
    /// A `Documentation` formatted as API reference.
    pub async fn generate_api_reference(&self, api_description: &str) -> Result<Documentation> {
        let start = Instant::now();

        if api_description.trim().is_empty() {
            return Err(Error::Synthesis(
                "Cannot generate API reference for empty content".to_string(),
            ));
        }

        let prompt = self.build_api_reference_prompt(api_description);
        let response = self.call_llm(&prompt).await?;
        let sections = self.parse_sections(&response);

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(Documentation {
            title: format!("API Reference: {}", self.extract_title(&response, api_description)),
            sections,
            format: self.config.format,
            generated_at: Utc::now(),
            version: self.config.version.clone(),
            audience: Audience::Developer,
            duration_ms,
            tokens_used: Self::estimate_tokens(api_description) + Self::estimate_tokens(&prompt),
        })
    }

    /// Generate a quick reference guide.
    ///
    /// Creates concise documentation focused on essential information
    /// for quick lookup.
    ///
    /// # Arguments
    ///
    /// * `topic` - The topic to create a quick reference for
    ///
    /// # Returns
    ///
    /// A concise `Documentation` for quick reference.
    pub async fn generate_quick_reference(&self, topic: &str) -> Result<Documentation> {
        let start = Instant::now();

        if topic.trim().is_empty() {
            return Err(Error::Synthesis(
                "Cannot generate quick reference for empty topic".to_string(),
            ));
        }

        let prompt = self.build_quick_reference_prompt(topic);
        let response = self.call_llm(&prompt).await?;
        let sections = self.parse_sections(&response);

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(Documentation {
            title: format!("Quick Reference: {}", topic),
            sections,
            format: self.config.format,
            generated_at: Utc::now(),
            version: self.config.version.clone(),
            audience: Audience::Expert,
            duration_ms,
            tokens_used: Self::estimate_tokens(topic) + Self::estimate_tokens(&prompt),
        })
    }

    /// Generate a single documentation section.
    ///
    /// # Arguments
    ///
    /// * `heading` - Section heading
    /// * `content` - Content to document
    ///
    /// # Returns
    ///
    /// A single `DocSection`.
    pub async fn generate_section(
        &self,
        heading: &str,
        content: &str,
    ) -> Result<DocSection> {
        if content.trim().is_empty() {
            return Err(Error::Synthesis(
                "Cannot generate section for empty content".to_string(),
            ));
        }

        let prompt = format!(
            r#"You are the Knowledge Recorder, a documentation specialist.

Generate documentation content for the following section.
Target audience: {} ({})

Section heading: {}

Content to document:
---
{}
---

Provide:
1. Clear, well-organized content
2. Examples if relevant ({}included)
3. Cross-references if applicable

Write only the section content, not the heading."#,
            self.config.target_audience.label(),
            self.config.target_audience.description(),
            heading,
            content,
            if self.config.include_examples { "" } else { "not " }
        );

        let response = self.call_llm(&prompt).await?;

        let mut section = DocSection::new(heading, response.trim());

        if self.config.include_examples {
            // Try to extract examples from the response
            let examples = self.extract_examples(&section.content);
            section.examples = examples;
        }

        Ok(section)
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /// Build the documentation prompt for the LLM.
    fn build_document_prompt(&self, content: &str) -> String {
        let audience_info = format!(
            "{} ({})",
            self.config.target_audience.label(),
            self.config.target_audience.description()
        );

        let example_instruction = if self.config.include_examples {
            "Include relevant code examples and usage demonstrations."
        } else {
            "Focus on conceptual explanation without code examples."
        };

        let depth_instruction = format!(
            "Use up to {} levels of headings for organization.",
            self.config.max_depth
        );

        format!(
            r#"You are the Knowledge Recorder, Agent 34 of Project Panpsychism.
Your role is to transform knowledge into well-structured, accessible documentation.

Following Spinoza's principles:
- CONATUS: Preserve the self-sustaining clarity of knowledge
- RATIO: Apply logical organization and structure
- LAETITIA: Enhance understanding through clear presentation
- NATURA: Respect the natural flow of information

Target Audience: {}
Output Format: {}
{}
{}

Content to document:
---
{}
---

Generate comprehensive documentation with:
1. A clear title (on the first line, preceded by #)
2. Organized sections with appropriate headings (## for main sections, ### for subsections)
3. Clear explanations appropriate for the target audience
4. {}
5. Cross-references where relevant (note: See also: ...)

Structure the output as Markdown with clear section breaks."#,
            audience_info,
            self.config.format.label(),
            depth_instruction,
            example_instruction,
            content,
            if self.config.include_api_reference {
                "API reference information if applicable"
            } else {
                "Practical usage information"
            }
        )
    }

    /// Build the API reference prompt.
    fn build_api_reference_prompt(&self, api_description: &str) -> String {
        format!(
            r#"You are the Knowledge Recorder, creating API reference documentation.

Generate comprehensive API reference documentation for:
---
{}
---

Structure the documentation with:
1. # API Title
2. ## Overview - Brief description of the API
3. ## Authentication - How to authenticate
4. ## Endpoints - Each endpoint with:
   - HTTP method and path
   - Description
   - Parameters (path, query, body)
   - Request/Response examples
   - Error codes
5. ## Error Handling - Common errors and solutions
6. ## Rate Limiting - If applicable

Use Markdown format with code blocks for examples.
Target audience: Software Developers"#,
            api_description
        )
    }

    /// Build the quick reference prompt.
    fn build_quick_reference_prompt(&self, topic: &str) -> String {
        format!(
            r#"You are the Knowledge Recorder, creating a quick reference guide.

Create a concise quick reference for: {}

Format requirements:
- Be extremely concise
- Use bullet points and tables
- Include only essential information
- Add brief code snippets where helpful
- Maximum 2-3 sections

Target audience: Experts who need quick lookup"#,
            topic
        )
    }

    /// Parse sections from LLM response.
    fn parse_sections(&self, response: &str) -> Vec<DocSection> {
        let mut sections = Vec::new();
        let mut current_heading = String::new();
        let mut current_level: u8 = 1;
        let mut current_content = String::new();

        for line in response.lines() {
            if line.starts_with('#') {
                // Save previous section
                if !current_heading.is_empty() && !current_content.trim().is_empty() {
                    let mut section =
                        DocSection::new(&current_heading, current_content.trim())
                            .with_level(current_level);

                    if self.config.include_examples {
                        section.examples = self.extract_examples(&section.content);
                    }

                    sections.push(section);
                }

                // Parse new heading
                let level = line.chars().take_while(|&c| c == '#').count() as u8;
                let heading = line.trim_start_matches('#').trim().to_string();

                current_heading = heading;
                current_level = level.min(self.config.max_depth);
                current_content = String::new();
            } else {
                current_content.push_str(line);
                current_content.push('\n');
            }
        }

        // Save last section
        if !current_heading.is_empty() && !current_content.trim().is_empty() {
            let mut section =
                DocSection::new(&current_heading, current_content.trim())
                    .with_level(current_level);

            if self.config.include_examples {
                section.examples = self.extract_examples(&section.content);
            }

            sections.push(section);
        }

        // If no sections were parsed, create a single section from the content
        if sections.is_empty() && !response.trim().is_empty() {
            sections.push(DocSection::new("Documentation", response.trim()));
        }

        sections
    }

    /// Extract code examples from content.
    fn extract_examples(&self, content: &str) -> Vec<String> {
        let mut examples = Vec::new();
        let mut in_code_block = false;
        let mut current_example = String::new();

        for line in content.lines() {
            if line.starts_with("```") {
                if in_code_block {
                    // End of code block
                    if !current_example.trim().is_empty() {
                        examples.push(current_example.trim().to_string());
                    }
                    current_example = String::new();
                    in_code_block = false;
                } else {
                    // Start of code block
                    in_code_block = true;
                }
            } else if in_code_block {
                current_example.push_str(line);
                current_example.push('\n');
            }
        }

        examples
    }

    /// Extract title from response or generate from content.
    fn extract_title(&self, response: &str, content: &str) -> String {
        // Try to find a title in the response
        for line in response.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("# ") {
                return trimmed.trim_start_matches('#').trim().to_string();
            }
        }

        // Generate a title from the content
        let words: Vec<&str> = content.split_whitespace().take(8).collect();
        if words.is_empty() {
            "Documentation".to_string()
        } else {
            format!("{}...", words.join(" "))
        }
    }

    /// Call the LLM with retry logic.
    async fn call_llm(&self, prompt: &str) -> Result<String> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let backoff = std::time::Duration::from_secs(1 << (attempt - 1));
                tokio::time::sleep(backoff).await;
            }

            let messages = vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }];

            match self.client.chat(messages).await {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        return Ok(choice.message.content.clone());
                    }
                    last_error = Some(Error::Synthesis("Empty response from LLM".to_string()));
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| Error::Synthesis("LLM call failed".to_string())))
    }

    /// Estimate token count (rough approximation: ~4 chars per token).
    fn estimate_tokens(text: &str) -> usize {
        text.len() / 4
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Escape HTML special characters.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DOC FORMAT TESTS
    // =========================================================================

    #[test]
    fn test_doc_format_extension() {
        assert_eq!(DocFormat::Markdown.extension(), "md");
        assert_eq!(DocFormat::Html.extension(), "html");
        assert_eq!(DocFormat::Pdf.extension(), "pdf");
        assert_eq!(DocFormat::Asciidoc.extension(), "adoc");
        assert_eq!(DocFormat::RestructuredText.extension(), "rst");
    }

    #[test]
    fn test_doc_format_mime_type() {
        assert_eq!(DocFormat::Markdown.mime_type(), "text/markdown");
        assert_eq!(DocFormat::Html.mime_type(), "text/html");
        assert_eq!(DocFormat::Pdf.mime_type(), "application/pdf");
    }

    #[test]
    fn test_doc_format_label() {
        assert_eq!(DocFormat::Markdown.label(), "Markdown");
        assert_eq!(DocFormat::Html.label(), "HTML");
        assert_eq!(DocFormat::Pdf.label(), "PDF");
        assert_eq!(DocFormat::Asciidoc.label(), "AsciiDoc");
        assert_eq!(DocFormat::RestructuredText.label(), "reStructuredText");
    }

    #[test]
    fn test_doc_format_supports_styling() {
        assert!(!DocFormat::Markdown.supports_styling());
        assert!(DocFormat::Html.supports_styling());
        assert!(DocFormat::Pdf.supports_styling());
        assert!(DocFormat::Asciidoc.supports_styling());
        assert!(!DocFormat::RestructuredText.supports_styling());
    }

    #[test]
    fn test_doc_format_is_plain_text() {
        assert!(DocFormat::Markdown.is_plain_text());
        assert!(!DocFormat::Html.is_plain_text());
        assert!(!DocFormat::Pdf.is_plain_text());
        assert!(DocFormat::Asciidoc.is_plain_text());
        assert!(DocFormat::RestructuredText.is_plain_text());
    }

    #[test]
    fn test_doc_format_display() {
        assert_eq!(format!("{}", DocFormat::Markdown), "Markdown");
        assert_eq!(format!("{}", DocFormat::Html), "HTML");
    }

    #[test]
    fn test_doc_format_from_str() {
        assert_eq!("markdown".parse::<DocFormat>().unwrap(), DocFormat::Markdown);
        assert_eq!("md".parse::<DocFormat>().unwrap(), DocFormat::Markdown);
        assert_eq!("html".parse::<DocFormat>().unwrap(), DocFormat::Html);
        assert_eq!("htm".parse::<DocFormat>().unwrap(), DocFormat::Html);
        assert_eq!("pdf".parse::<DocFormat>().unwrap(), DocFormat::Pdf);
        assert_eq!("asciidoc".parse::<DocFormat>().unwrap(), DocFormat::Asciidoc);
        assert_eq!("adoc".parse::<DocFormat>().unwrap(), DocFormat::Asciidoc);
        assert_eq!("rst".parse::<DocFormat>().unwrap(), DocFormat::RestructuredText);
    }

    #[test]
    fn test_doc_format_from_str_invalid() {
        let result = "invalid".parse::<DocFormat>();
        assert!(result.is_err());
    }

    #[test]
    fn test_doc_format_default() {
        assert_eq!(DocFormat::default(), DocFormat::Markdown);
    }

    // =========================================================================
    // AUDIENCE TESTS
    // =========================================================================

    #[test]
    fn test_audience_description() {
        assert!(Audience::Developer.description().contains("developers"));
        assert!(Audience::EndUser.description().contains("end users"));
        assert!(Audience::Administrator.description().contains("administrators"));
        assert!(Audience::Beginner.description().contains("Newcomers"));
        assert!(Audience::Expert.description().contains("Advanced"));
    }

    #[test]
    fn test_audience_technical_level() {
        assert_eq!(Audience::Beginner.technical_level(), 0);
        assert_eq!(Audience::EndUser.technical_level(), 1);
        assert_eq!(Audience::Administrator.technical_level(), 2);
        assert_eq!(Audience::Developer.technical_level(), 3);
        assert_eq!(Audience::Expert.technical_level(), 4);
    }

    #[test]
    fn test_audience_expects_code_examples() {
        assert!(Audience::Developer.expects_code_examples());
        assert!(Audience::Expert.expects_code_examples());
        assert!(!Audience::EndUser.expects_code_examples());
        assert!(!Audience::Beginner.expects_code_examples());
        assert!(!Audience::Administrator.expects_code_examples());
    }

    #[test]
    fn test_audience_needs_detailed_explanations() {
        assert!(Audience::Beginner.needs_detailed_explanations());
        assert!(Audience::EndUser.needs_detailed_explanations());
        assert!(!Audience::Developer.needs_detailed_explanations());
        assert!(!Audience::Expert.needs_detailed_explanations());
    }

    #[test]
    fn test_audience_display() {
        assert_eq!(format!("{}", Audience::Developer), "Developer");
        assert_eq!(format!("{}", Audience::EndUser), "End User");
    }

    #[test]
    fn test_audience_from_str() {
        assert_eq!("developer".parse::<Audience>().unwrap(), Audience::Developer);
        assert_eq!("dev".parse::<Audience>().unwrap(), Audience::Developer);
        assert_eq!("enduser".parse::<Audience>().unwrap(), Audience::EndUser);
        assert_eq!("user".parse::<Audience>().unwrap(), Audience::EndUser);
        assert_eq!("admin".parse::<Audience>().unwrap(), Audience::Administrator);
        assert_eq!("beginner".parse::<Audience>().unwrap(), Audience::Beginner);
        assert_eq!("expert".parse::<Audience>().unwrap(), Audience::Expert);
    }

    #[test]
    fn test_audience_from_str_invalid() {
        let result = "invalid".parse::<Audience>();
        assert!(result.is_err());
    }

    #[test]
    fn test_audience_default() {
        assert_eq!(Audience::default(), Audience::Developer);
    }

    // =========================================================================
    // DOC SECTION TESTS
    // =========================================================================

    #[test]
    fn test_doc_section_creation() {
        let section = DocSection::new("Getting Started", "This is the content.");
        assert_eq!(section.heading, "Getting Started");
        assert_eq!(section.content, "This is the content.");
        assert_eq!(section.level, 1);
        assert!(section.examples.is_empty());
        assert!(section.cross_refs.is_empty());
    }

    #[test]
    fn test_doc_section_builder() {
        let section = DocSection::new("Test", "Content")
            .with_level(2)
            .with_example("print('hello')")
            .with_cross_ref("See also: Other Section");

        assert_eq!(section.level, 2);
        assert_eq!(section.examples.len(), 1);
        assert_eq!(section.cross_refs.len(), 1);
    }

    #[test]
    fn test_doc_section_level_clamping() {
        let section = DocSection::new("Test", "Content").with_level(10);
        assert_eq!(section.level, 6);

        let section = DocSection::new("Test", "Content").with_level(0);
        assert_eq!(section.level, 1);
    }

    #[test]
    fn test_doc_section_multiple_examples() {
        let section = DocSection::new("Test", "Content")
            .with_examples(vec!["example1".to_string(), "example2".to_string()])
            .with_example("example3");

        assert_eq!(section.examples.len(), 3);
    }

    #[test]
    fn test_doc_section_multiple_cross_refs() {
        let section = DocSection::new("Test", "Content")
            .with_cross_refs(vec!["ref1".to_string(), "ref2".to_string()])
            .with_cross_ref("ref3");

        assert_eq!(section.cross_refs.len(), 3);
    }

    #[test]
    fn test_doc_section_has_examples() {
        let section = DocSection::new("Test", "Content");
        assert!(!section.has_examples());

        let section = section.with_example("example");
        assert!(section.has_examples());
    }

    #[test]
    fn test_doc_section_has_cross_refs() {
        let section = DocSection::new("Test", "Content");
        assert!(!section.has_cross_refs());

        let section = section.with_cross_ref("ref");
        assert!(section.has_cross_refs());
    }

    #[test]
    fn test_doc_section_word_count() {
        let section = DocSection::new("Test", "one two three four five");
        assert_eq!(section.word_count(), 5);

        let section = DocSection::new("Test", "");
        assert_eq!(section.word_count(), 0);
    }

    #[test]
    fn test_doc_section_render_markdown() {
        let section = DocSection::new("Test Heading", "Test content.")
            .with_level(2)
            .with_example("code example");

        let rendered = section.render_markdown();
        assert!(rendered.contains("## Test Heading"));
        assert!(rendered.contains("Test content."));
        assert!(rendered.contains("```\ncode example\n```"));
    }

    #[test]
    fn test_doc_section_render_html() {
        let section = DocSection::new("Test Heading", "Test content.")
            .with_level(2)
            .with_example("code");

        let rendered = section.render_html();
        assert!(rendered.contains("<h2>Test Heading</h2>"));
        assert!(rendered.contains("Test content."));
        assert!(rendered.contains("<pre><code>code</code></pre>"));
    }

    #[test]
    fn test_doc_section_render_asciidoc() {
        let section = DocSection::new("Test", "Content").with_level(1);
        let rendered = section.render_asciidoc();
        assert!(rendered.contains("== Test"));
    }

    #[test]
    fn test_doc_section_render_rst() {
        let section = DocSection::new("Test", "Content").with_level(1);
        let rendered = section.render_rst();
        assert!(rendered.contains("Test\n===="));
    }

    #[test]
    fn test_doc_section_display() {
        let section = DocSection::new("Test", "Content");
        let display = format!("{}", section);
        assert!(display.contains("# Test"));
        assert!(display.contains("Content"));
    }

    // =========================================================================
    // DOCUMENTATION TESTS
    // =========================================================================

    #[test]
    fn test_documentation_creation() {
        let doc = Documentation::new("Test Documentation");
        assert_eq!(doc.title, "Test Documentation");
        assert!(doc.sections.is_empty());
        assert_eq!(doc.format, DocFormat::Markdown);
        assert_eq!(doc.version, "1.0.0");
    }

    #[test]
    fn test_documentation_builder() {
        let doc = Documentation::new("Test")
            .with_section(DocSection::new("Section 1", "Content 1"))
            .with_format(DocFormat::Html)
            .with_version("2.0.0")
            .with_audience(Audience::Expert);

        assert_eq!(doc.section_count(), 1);
        assert_eq!(doc.format, DocFormat::Html);
        assert_eq!(doc.version, "2.0.0");
        assert_eq!(doc.audience, Audience::Expert);
    }

    #[test]
    fn test_documentation_multiple_sections() {
        let doc = Documentation::new("Test")
            .with_sections(vec![
                DocSection::new("Section 1", "Content 1"),
                DocSection::new("Section 2", "Content 2"),
            ]);

        assert_eq!(doc.section_count(), 2);
    }

    #[test]
    fn test_documentation_word_count() {
        let doc = Documentation::new("Test")
            .with_section(DocSection::new("S1", "one two three"))
            .with_section(DocSection::new("S2", "four five"));

        assert_eq!(doc.word_count(), 5);
    }

    #[test]
    fn test_documentation_example_count() {
        let doc = Documentation::new("Test")
            .with_section(DocSection::new("S1", "Content").with_example("ex1"))
            .with_section(DocSection::new("S2", "Content").with_examples(vec![
                "ex2".to_string(),
                "ex3".to_string(),
            ]));

        assert_eq!(doc.example_count(), 3);
    }

    #[test]
    fn test_documentation_table_of_contents() {
        let doc = Documentation::new("Test")
            .with_section(DocSection::new("First", "Content").with_level(1))
            .with_section(DocSection::new("Second", "Content").with_level(2));

        let toc = doc.table_of_contents();
        assert_eq!(toc.len(), 2);
        assert_eq!(toc[0], (1, "First".to_string()));
        assert_eq!(toc[1], (2, "Second".to_string()));
    }

    #[test]
    fn test_documentation_render_markdown() {
        let doc = Documentation::new("Test Doc")
            .with_section(DocSection::new("Section", "Content").with_level(2));

        let rendered = doc.render_markdown();
        assert!(rendered.contains("# Test Doc"));
        assert!(rendered.contains("## Section"));
        assert!(rendered.contains("Content"));
    }

    #[test]
    fn test_documentation_render_html() {
        let doc = Documentation::new("Test Doc")
            .with_section(DocSection::new("Section", "Content"));

        let rendered = doc.render_html();
        assert!(rendered.contains("<!DOCTYPE html>"));
        assert!(rendered.contains("<h1>Test Doc</h1>"));
        assert!(rendered.contains("<h1>Section</h1>"));
    }

    #[test]
    fn test_documentation_render_pdf() {
        let doc = Documentation::new("Test")
            .with_format(DocFormat::Pdf)
            .with_section(DocSection::new("Section", "Content"));

        let rendered = doc.render();
        assert!(rendered.contains("title: \"Test\""));
        assert!(rendered.contains("documentclass: article"));
    }

    #[test]
    fn test_documentation_render_asciidoc() {
        let doc = Documentation::new("Test")
            .with_format(DocFormat::Asciidoc)
            .with_section(DocSection::new("Section", "Content"));

        let rendered = doc.render();
        assert!(rendered.contains("= Test"));
        assert!(rendered.contains(":toc:"));
    }

    #[test]
    fn test_documentation_render_rst() {
        let doc = Documentation::new("Test")
            .with_format(DocFormat::RestructuredText)
            .with_section(DocSection::new("Section", "Content"));

        let rendered = doc.render();
        assert!(rendered.contains("Test\n===="));
        assert!(rendered.contains(".. contents::"));
    }

    #[test]
    fn test_documentation_stats() {
        let doc = Documentation::new("Test")
            .with_section(DocSection::new("S1", "one two three"));

        let stats = doc.stats();
        assert!(stats.contains("1 sections"));
        assert!(stats.contains("3 words"));
    }

    #[test]
    fn test_documentation_display() {
        let doc = Documentation::new("Test")
            .with_section(DocSection::new("Section", "Content"));

        let display = format!("{}", doc);
        assert!(display.contains("# Test"));
    }

    // =========================================================================
    // DOC CONFIG TESTS
    // =========================================================================

    #[test]
    fn test_doc_config_default() {
        let config = DocConfig::default();
        assert!(config.include_examples);
        assert!(!config.include_api_reference);
        assert_eq!(config.max_depth, 3);
        assert_eq!(config.target_audience, Audience::Developer);
        assert_eq!(config.format, DocFormat::Markdown);
        assert!(config.include_toc);
        assert!(config.include_cross_refs);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.version, "1.0.0");
    }

    #[test]
    fn test_doc_config_for_api() {
        let config = DocConfig::for_api();
        assert!(config.include_examples);
        assert!(config.include_api_reference);
        assert_eq!(config.target_audience, Audience::Developer);
    }

    #[test]
    fn test_doc_config_for_users() {
        let config = DocConfig::for_users();
        assert!(config.include_examples);
        assert!(!config.include_api_reference);
        assert_eq!(config.target_audience, Audience::EndUser);
    }

    #[test]
    fn test_doc_config_for_quick_reference() {
        let config = DocConfig::for_quick_reference();
        assert_eq!(config.max_depth, 2);
        assert!(!config.include_toc);
        assert_eq!(config.target_audience, Audience::Expert);
    }

    // =========================================================================
    // DOCUMENTER AGENT BUILDER TESTS
    // =========================================================================

    #[test]
    fn test_documenter_builder() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgentBuilder::new(client)
            .with_format(DocFormat::Html)
            .with_audience(Audience::Expert)
            .with_examples(false)
            .with_api_reference(true)
            .with_max_depth(4)
            .with_toc(false)
            .with_cross_refs(false)
            .with_max_retries(5)
            .with_version("2.0.0")
            .build();

        assert_eq!(agent.config().format, DocFormat::Html);
        assert_eq!(agent.config().target_audience, Audience::Expert);
        assert!(!agent.config().include_examples);
        assert!(agent.config().include_api_reference);
        assert_eq!(agent.config().max_depth, 4);
        assert!(!agent.config().include_toc);
        assert!(!agent.config().include_cross_refs);
        assert_eq!(agent.config().max_retries, 5);
        assert_eq!(agent.config().version, "2.0.0");
    }

    #[test]
    fn test_documenter_builder_with_config() {
        let client = Arc::new(GeminiClient::new());
        let config = DocConfig::for_api();
        let agent = DocumenterAgentBuilder::new(client)
            .with_config(config)
            .build();

        assert!(agent.config().include_api_reference);
    }

    // =========================================================================
    // DOCUMENTER AGENT TESTS
    // =========================================================================

    #[test]
    fn test_documenter_agent_creation() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);
        assert_eq!(agent.config().format, DocFormat::Markdown);
    }

    #[test]
    fn test_documenter_agent_with_config() {
        let client = Arc::new(GeminiClient::new());
        let config = DocConfig {
            format: DocFormat::Html,
            ..Default::default()
        };
        let agent = DocumenterAgent::with_config(client, config);
        assert_eq!(agent.config().format, DocFormat::Html);
    }

    #[test]
    fn test_documenter_agent_builder_method() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::builder(client)
            .with_format(DocFormat::Asciidoc)
            .build();
        assert_eq!(agent.config().format, DocFormat::Asciidoc);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(DocumenterAgent::estimate_tokens("12345678"), 2);
        assert_eq!(DocumenterAgent::estimate_tokens(""), 0);
    }

    #[test]
    fn test_extract_examples() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let content = r#"Some text
```
example code
```
More text
```python
another example
```"#;

        let examples = agent.extract_examples(content);
        assert_eq!(examples.len(), 2);
        assert!(examples[0].contains("example code"));
        assert!(examples[1].contains("another example"));
    }

    #[test]
    fn test_extract_examples_empty() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let examples = agent.extract_examples("No code blocks here");
        assert!(examples.is_empty());
    }

    #[test]
    fn test_extract_title_from_response() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let response = "# My Documentation Title\n\nSome content here.";
        let title = agent.extract_title(response, "fallback");
        assert_eq!(title, "My Documentation Title");
    }

    #[test]
    fn test_extract_title_fallback() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let response = "No heading here.";
        let title = agent.extract_title(response, "Some content to use as fallback");
        assert!(title.contains("Some content to use"));
        assert!(title.ends_with("..."));
    }

    #[test]
    fn test_parse_sections() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let response = r#"# Title
Intro content

## Section One
Content for section one

### Subsection
Subsection content

## Section Two
Content for section two"#;

        let sections = agent.parse_sections(response);
        assert_eq!(sections.len(), 4);
        assert_eq!(sections[0].heading, "Title");
        assert_eq!(sections[0].level, 1);
        assert_eq!(sections[1].heading, "Section One");
        assert_eq!(sections[1].level, 2);
        assert_eq!(sections[2].heading, "Subsection");
        assert_eq!(sections[2].level, 3);
    }

    #[test]
    fn test_parse_sections_empty() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let sections = agent.parse_sections("");
        assert!(sections.is_empty());
    }

    #[test]
    fn test_parse_sections_no_headings() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let sections = agent.parse_sections("Just plain text without headings");
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].heading, "Documentation");
    }

    #[tokio::test]
    async fn test_document_empty_content() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let result = agent.document("").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty content"));
    }

    #[tokio::test]
    async fn test_document_whitespace_only() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let result = agent.document("   \n\t  ").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_api_reference_empty() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let result = agent.generate_api_reference("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_quick_reference_empty() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let result = agent.generate_quick_reference("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_section_empty() {
        let client = Arc::new(GeminiClient::new());
        let agent = DocumenterAgent::new(client);

        let result = agent.generate_section("Heading", "").await;
        assert!(result.is_err());
    }

    // =========================================================================
    // UTILITY FUNCTION TESTS
    // =========================================================================

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<div>"), "&lt;div&gt;");
        assert_eq!(html_escape("&"), "&amp;");
        assert_eq!(html_escape("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(html_escape("'single'"), "&#39;single&#39;");
        assert_eq!(html_escape("normal text"), "normal text");
    }

    #[test]
    fn test_html_escape_complex() {
        assert_eq!(
            html_escape("<script>alert('xss')</script>"),
            "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;"
        );
    }
}
