/**
 * Main Application for Panpsychism Dashboard
 */

const App = {
    /**
     * Initialize the dashboard
     */
    init() {
        console.log('🧠 Initializing Panpsychism Dashboard...');

        // Load system metrics
        this.loadSystemMetrics();

        // Load agent list
        this.loadAgentList();

        // Initialize all charts
        Charts.initAll();

        // Start auto-refresh (every 5 seconds)
        this.startAutoRefresh();

        console.log('✅ Dashboard initialized successfully');
    },

    /**
     * Load and display system metrics
     */
    loadSystemMetrics() {
        const metrics = API.getSystemMetrics();

        // Update overview cards
        document.getElementById('totalAgents').textContent = metrics.totalAgents;
        document.getElementById('activeAgents').textContent = metrics.activeAgents;
        document.getElementById('uptime').textContent = `${metrics.uptimePercent}%`;
        document.getElementById('memory').textContent = `${metrics.memoryUsed} GB`;

        // Update progress bar
        const progressBar = document.querySelector('.progress-fill');
        if (progressBar) {
            progressBar.style.width = `${metrics.memoryPercent}%`;
        }
    },

    /**
     * Load and display agent list
     */
    loadAgentList() {
        const agents = API.getAllAgents();
        const agentList = document.getElementById('agentList');

        if (!agentList) return;

        agentList.innerHTML = agents.map(agent => `
            <div class="agent-item">
                <div class="agent-status ${agent.active ? 'active' : 'inactive'}"></div>
                <div class="agent-name">${agent.name}</div>
                <div class="agent-tier">T${agent.tier}</div>
            </div>
        `).join('');
    },

    /**
     * Update dynamic metrics
     */
    updateMetrics() {
        const metrics = API.getSystemMetrics();

        // Animate metric value changes
        this.animateValue('activeAgents', parseInt(document.getElementById('activeAgents').textContent), metrics.activeAgents, 500);
        this.animateValue('uptime', parseFloat(document.getElementById('uptime').textContent), metrics.uptimePercent, 500);
    },

    /**
     * Animate numeric value changes
     */
    animateValue(elementId, start, end, duration) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const range = end - start;
        const increment = range / (duration / 16);
        let current = start;

        const timer = setInterval(() => {
            current += increment;
            if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                current = end;
                clearInterval(timer);
            }

            // Format based on value type
            if (elementId === 'uptime') {
                element.textContent = `${current.toFixed(1)}%`;
            } else {
                element.textContent = Math.round(current);
            }
        }, 16);
    },

    /**
     * Start auto-refresh interval
     */
    startAutoRefresh() {
        // Refresh metrics every 5 seconds
        setInterval(() => {
            this.updateMetrics();
            console.log('🔄 Metrics refreshed');
        }, 5000);

        // Refresh RPM chart every 30 seconds
        setInterval(() => {
            const rpmCanvas = document.getElementById('rpmChart');
            if (rpmCanvas) {
                const existingChart = Chart.getChart(rpmCanvas);
                if (existingChart) {
                    existingChart.destroy();
                }
                Charts.createRPMChart('rpmChart');
                console.log('📊 RPM chart refreshed');
            }
        }, 30000);
    },

    /**
     * Handle errors
     */
    handleError(error) {
        console.error('❌ Dashboard Error:', error);
        // Could show error notification to user
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        App.init();
    } catch (error) {
        App.handleError(error);
    }
});

// Handle visibility change (pause/resume when tab is hidden/visible)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('⏸️ Dashboard paused (tab hidden)');
    } else {
        console.log('▶️ Dashboard resumed (tab visible)');
        App.updateMetrics();
    }
});

// Export for debugging in console
window.PanpsychismDashboard = App;
