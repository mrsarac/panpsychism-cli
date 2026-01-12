/**
 * API Client for Panpsychism Dashboard
 * Provides mock data for demonstration purposes
 */

const API = {
    /**
     * Get system overview metrics
     */
    getSystemMetrics() {
        return {
            totalAgents: 40,
            activeAgents: 37,
            uptime: "7d 14h 23m",
            uptimePercent: 99.8,
            memoryUsed: 2.4,
            memoryTotal: 8.0,
            memoryPercent: 30
        };
    },

    /**
     * Get agent distribution by tier
     */
    getAgentsByTier() {
        return [
            { tier: "Tier 1: Interpreters", count: 5, color: "#fbbf24" },
            { tier: "Tier 2: Creators", count: 5, color: "#f59e0b" },
            { tier: "Tier 3: Alchemists", count: 5, color: "#d97706" },
            { tier: "Tier 4: Oracles", count: 5, color: "#b45309" },
            { tier: "Tier 5: Integrators", count: 5, color: "#92400e" },
            { tier: "Tier 6: Guardians", count: 5, color: "#78350f" },
            { tier: "Tier 7: Architects", count: 5, color: "#451a03" },
            { tier: "Tier 8: Navigators", count: 5, color: "#3b82f6" }
        ];
    },

    /**
     * Get all agents with status
     */
    getAllAgents() {
        const agents = [
            // Tier 1: Interpreters
            { name: "Analyzer", tier: 1, active: true },
            { name: "Parser", tier: 1, active: true },
            { name: "Extractor", tier: 1, active: true },
            { name: "Classifier", tier: 1, active: true },
            { name: "Validator", tier: 1, active: true },

            // Tier 2: Creators
            { name: "Writer", tier: 2, active: true },
            { name: "Generator", tier: 2, active: true },
            { name: "Composer", tier: 2, active: true },
            { name: "Designer", tier: 2, active: true },
            { name: "Builder", tier: 2, active: true },

            // Tier 3: Alchemists
            { name: "Synthesizer", tier: 3, active: true },
            { name: "Contextualizer", tier: 3, active: true },
            { name: "Formatter", tier: 3, active: true },
            { name: "Summarizer", tier: 3, active: true },
            { name: "Expander", tier: 3, active: true },

            // Tier 4: Oracles
            { name: "Predictor", tier: 4, active: true },
            { name: "Recommender", tier: 4, active: true },
            { name: "Evaluator", tier: 4, active: true },
            { name: "Debugger", tier: 4, active: false },
            { name: "Learner", tier: 4, active: true },

            // Tier 5: Integrators
            { name: "Aggregator", tier: 5, active: true },
            { name: "Orchestrator", tier: 5, active: true },
            { name: "Coordinator", tier: 5, active: true },
            { name: "Merger", tier: 5, active: true },
            { name: "Unifier", tier: 5, active: true },

            // Tier 6: Guardians
            { name: "Validator", tier: 6, active: true },
            { name: "Auditor", tier: 6, active: true },
            { name: "Monitor", tier: 6, active: true },
            { name: "Sentinel", tier: 6, active: false },
            { name: "Inspector", tier: 6, active: true },

            // Tier 7: Architects
            { name: "Planner", tier: 7, active: true },
            { name: "Strategist", tier: 7, active: true },
            { name: "Optimizer", tier: 7, active: true },
            { name: "Refiner", tier: 7, active: true },
            { name: "Evolver", tier: 7, active: false },

            // Tier 8: Navigators
            { name: "Router", tier: 8, active: true },
            { name: "Director", tier: 8, active: true },
            { name: "Guide", tier: 8, active: true },
            { name: "Pathfinder", tier: 8, active: true },
            { name: "Navigator", tier: 8, active: true }
        ];

        return agents;
    },

    /**
     * Get latency histogram data
     */
    getLatencyData() {
        return {
            labels: ['0-50ms', '50-100ms', '100-200ms', '200-500ms', '500ms+'],
            data: [450, 320, 180, 40, 10]
        };
    },

    /**
     * Get requests per minute over time
     */
    getRPMData() {
        const now = new Date();
        const labels = [];
        const data = [];

        for (let i = 29; i >= 0; i--) {
            const time = new Date(now - i * 60000);
            labels.push(time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
            data.push(Math.floor(Math.random() * 50 + 30));
        }

        return { labels, data };
    },

    /**
     * Get LLM provider distribution
     */
    getProviderData() {
        return [
            { provider: "OpenAI", percentage: 35, color: "#10b981" },
            { provider: "Anthropic", percentage: 30, color: "#3b82f6" },
            { provider: "Ollama (Local)", percentage: 20, color: "#8b5cf6" },
            { provider: "Google Gemini", percentage: 10, color: "#ef4444" },
            { provider: "Others", percentage: 5, color: "#6b7280" }
        ];
    },

    /**
     * Get token usage stats
     */
    getTokenStats() {
        return {
            inputTokens: 1234567,
            outputTokens: 567890,
            totalTokens: 1802457
        };
    },

    /**
     * Get cost tracking data
     */
    getCostData() {
        return {
            today: 12.45,
            week: 87.32,
            month: 342.18
        };
    },

    /**
     * Get error rate stats
     */
    getErrorStats() {
        return {
            errorRate: 0.2,
            errorCount: 2,
            totalRequests: 1000
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = API;
}
