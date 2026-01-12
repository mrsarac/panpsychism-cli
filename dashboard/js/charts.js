/**
 * Chart.js Configuration for Panpsychism Dashboard
 */

const Charts = {
    /**
     * Default chart options
     */
    defaultOptions: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#c9d1d9',
                    font: {
                        family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'
                    }
                }
            }
        },
        scales: {
            x: {
                ticks: { color: '#8b949e' },
                grid: { color: '#30363d' }
            },
            y: {
                ticks: { color: '#8b949e' },
                grid: { color: '#30363d' }
            }
        }
    },

    /**
     * Create agent tier distribution pie chart
     */
    createTierChart(canvasId) {
        const data = API.getAgentsByTier();
        const ctx = document.getElementById(canvasId);

        return new Chart(ctx, {
            type: 'pie',
            data: {
                labels: data.map(d => d.tier),
                datasets: [{
                    data: data.map(d => d.count),
                    backgroundColor: data.map(d => d.color),
                    borderColor: '#21262d',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: '#c9d1d9',
                            font: {
                                size: 11,
                                family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'
                            },
                            boxWidth: 15,
                            padding: 10
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed} agents`;
                            }
                        }
                    }
                }
            }
        });
    },

    /**
     * Create latency histogram
     */
    createLatencyChart(canvasId) {
        const data = API.getLatencyData();
        const ctx = document.getElementById(canvasId);

        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Request Count',
                    data: data.data,
                    backgroundColor: 'rgba(251, 191, 36, 0.6)',
                    borderColor: '#fbbf24',
                    borderWidth: 2,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#21262d',
                        titleColor: '#fbbf24',
                        bodyColor: '#c9d1d9',
                        borderColor: '#30363d',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#8b949e' },
                        grid: { display: false }
                    },
                    y: {
                        ticks: { color: '#8b949e' },
                        grid: { color: '#30363d' },
                        beginAtZero: true
                    }
                }
            }
        });
    },

    /**
     * Create LLM provider distribution chart
     */
    createProviderChart(canvasId) {
        const data = API.getProviderData();
        const ctx = document.getElementById(canvasId);

        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.map(d => d.provider),
                datasets: [{
                    data: data.map(d => d.percentage),
                    backgroundColor: data.map(d => d.color),
                    borderColor: '#21262d',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#c9d1d9',
                            font: {
                                size: 11,
                                family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'
                            },
                            boxWidth: 15,
                            padding: 10
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed}%`;
                            }
                        }
                    }
                },
                cutout: '60%'
            }
        });
    },

    /**
     * Create requests per minute line chart
     */
    createRPMChart(canvasId) {
        const data = API.getRPMData();
        const ctx = document.getElementById(canvasId);

        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Requests/min',
                    data: data.data,
                    borderColor: '#fbbf24',
                    backgroundColor: 'rgba(251, 191, 36, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#fbbf24',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#21262d',
                        titleColor: '#fbbf24',
                        bodyColor: '#c9d1d9',
                        borderColor: '#30363d',
                        borderWidth: 1,
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#8b949e',
                            maxRotation: 45,
                            minRotation: 45
                        },
                        grid: { display: false }
                    },
                    y: {
                        ticks: { color: '#8b949e' },
                        grid: { color: '#30363d' },
                        beginAtZero: true
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    },

    /**
     * Initialize all charts
     */
    initAll() {
        this.createTierChart('tierChart');
        this.createLatencyChart('latencyChart');
        this.createProviderChart('providerChart');
        this.createRPMChart('rpmChart');
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Charts;
}
