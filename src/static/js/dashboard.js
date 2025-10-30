// API Base URL
const API_BASE = '';

// DOM Elements
let sentimentChart, ratingChart, issuesChart, trendsChart;

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', function() {
    // GSAP Animations on page load
    animatePageLoad();
    initializeDashboard();
});

// Page load animations with GSAP
function animatePageLoad() {
    // Set initial state for elements to ensure they're visible
    gsap.set('.page-header, .stat-card, .chart-card', { opacity: 1 , y: 2});

    // Animate page header
    gsap.from('.page-header', {
        duration: 0.8,
        y: -30,
        opacity: 0,
        ease: 'power3.out'
    });

    // Animate stat cards with stagger
    gsap.from('.stat-card', {
        duration: 0.8,
        y: 50,
        opacity: 0,
        stagger: 0.1,
        ease: 'power3.out',
        delay: 0.2
    });

    // Animate chart cards
    gsap.from('.chart-card', {
        duration: 1,
        y: 60,
        opacity: 0,
        stagger: 0.15,
        ease: 'power3.out',
        delay: 0.5
    });

    // Add hover effects to stat cards
    document.querySelectorAll('.stat-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            gsap.to(card, {
                duration: 0.3,
                y: -5,
                boxShadow: '0 8px 20px rgba(0, 0, 0, 0.15)',
                ease: 'power2.out'
            });
        });

        card.addEventListener('mouseleave', () => {
            gsap.to(card, {
                duration: 0.3,
                y: 0,
                boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
                ease: 'power2.out'
            });
        });
    });
}

async function initializeDashboard() {
    try {
        await Promise.all([
            loadStatistics(),
            loadIssues(),
            loadTrends()
        ]);
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        showError('Failed to load dashboard data');
    }
}

// Load Statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();
        
        // Update stat cards
        updateStatCard('total-feedback', data.total_feedback);
        updateStatCard('avg-sentiment', data.average_rating.toFixed(1) + '/5');
        updateStatCard('new-issues', data.total_negative || 0);
        
        // Calculate resolution rate
        const resolutionRate = ((data.total_positive / data.total_feedback) * 100).toFixed(0);
        updateStatCard('resolution-rate', resolutionRate + '%');
        
        // Update stat changes
        updateStatChange('total-feedback', '+5.4%', 'positive');
        updateStatChange('avg-sentiment', '-0.1%', 'negative');
        updateStatChange('new-issues', '+12%', 'positive');
        updateStatChange('resolution-rate', '+1.3%', 'positive');
        
        // Create charts
        createSentimentChart(data.sentiment_distribution);
        createRatingChart(data.rating_distribution);
        
    } catch (error) {
        console.error('Error loading statistics:', error);
        throw error;
    }
}

// Load Issues
async function loadIssues() {
    try {
        const response = await fetch(`${API_BASE}/api/issues`);
        const data = await response.json();
        
        createIssuesChart(data.categories, data.counts);
        
    } catch (error) {
        console.error('Error loading issues:', error);
        throw error;
    }
}

// Load Trends
async function loadTrends() {
    try {
        const response = await fetch(`${API_BASE}/api/trends`);
        const data = await response.json();
        
        createTrendsChart(data.labels, data.positive, data.negative, data.neutral);
        
    } catch (error) {
        console.error('Error loading trends:', error);
        throw error;
    }
}

// Predict Sentiment
async function predictSentiment() {
    const textarea = document.getElementById('feedback-text');
    const text = textarea.value.trim();
    
    if (!text) {
        showError('Please enter customer feedback');
        return;
    }
    
    // Show loading
    const resultDiv = document.getElementById('prediction-result');
    const emptyDiv = document.getElementById('prediction-empty');
    
    emptyDiv.innerHTML = '<div class="spinner"></div>';
    
    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        // Hide empty state
        emptyDiv.classList.add('hidden');
        resultDiv.classList.remove('hidden');
        
        // Update sentiment
        const sentimentValue = document.getElementById('sentiment-value');
        sentimentValue.textContent = data.prediction.charAt(0).toUpperCase() + data.prediction.slice(1);
        sentimentValue.className = 'sentiment-value ' + data.prediction;
        
        // Update confidence
        const confidenceFill = document.getElementById('confidence-fill');
        const confidenceText = document.getElementById('confidence-text');
        const confidencePercent = (data.confidence * 100).toFixed(0);
        
        confidenceFill.style.width = confidencePercent + '%';
        confidenceText.textContent = 'Confidence: ' + confidencePercent + '%';
        
        // Update probabilities
        document.getElementById('prob-positive').textContent = (data.probabilities.positive * 100).toFixed(1) + '%';
        document.getElementById('prob-negative').textContent = (data.probabilities.negative * 100).toFixed(1) + '%';
        document.getElementById('prob-neutral').textContent = (data.probabilities.neutral * 100).toFixed(1) + '%';
        
    } catch (error) {
        console.error('Error predicting sentiment:', error);
        emptyDiv.classList.remove('hidden');
        emptyDiv.innerHTML = '<p class="prediction-empty">Error analyzing sentiment</p>';
        showError('Failed to analyze sentiment');
    }
}

// Update Stat Card
function updateStatCard(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = typeof value === 'number' ? value.toLocaleString() : value;
    }
}

// Update Stat Change
function updateStatChange(id, value, type) {
    const element = document.getElementById(id + '-change');
    if (element) {
        element.textContent = value;
        element.className = 'stat-change ' + type;
    }
}

// Create Sentiment Chart (Doughnut)
function createSentimentChart(distribution) {
    const ctx = document.getElementById('sentiment-chart');
    if (!ctx) return;
    
    if (sentimentChart) {
        sentimentChart.destroy();
    }
    
    sentimentChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Negative', 'Neutral'],
            datasets: [{
                data: [
                    distribution.positive || 0,
                    distribution.negative || 0,
                    distribution.neutral || 0
                ],
                backgroundColor: [
                    '#10b981',
                    '#ef4444',
                    '#f59e0b'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 13,
                            family: 'Inter'
                        }
                    }
                }
            },
            cutout: '70%'
        }
    });
}

// Create Rating Chart (Bar)
function createRatingChart(distribution) {
    const ctx = document.getElementById('rating-chart');
    if (!ctx) return;
    
    if (ratingChart) {
        ratingChart.destroy();
    }
    
    const ratings = [1, 2, 3, 4, 5];
    const counts = ratings.map(r => distribution[r] || 0);
    
    ratingChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ratings.map(r => r + ' ⭐'),
            datasets: [{
                label: 'Count',
                data: counts,
                backgroundColor: '#6366f1',
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#f3f4f6'
                    },
                    ticks: {
                        font: {
                            family: 'Inter'
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            family: 'Inter'
                        }
                    }
                }
            }
        }
    });
}

// Create Issues Chart (Horizontal Bar)
function createIssuesChart(categories, counts) {
    const ctx = document.getElementById('issues-chart');
    if (!ctx) return;
    
    if (issuesChart) {
        issuesChart.destroy();
    }
    
    issuesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: categories,
            datasets: [{
                label: 'Mentions',
                data: counts,
                backgroundColor: '#06b6d4',
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        color: '#f3f4f6'
                    },
                    ticks: {
                        font: {
                            family: 'Inter'
                        }
                    }
                },
                y: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            family: 'Inter',
                            size: 12
                        }
                    }
                }
            }
        }
    });
}

// Create Trends Chart (Stacked Bar)
function createTrendsChart(labels, positive, negative, neutral) {
    const ctx = document.getElementById('trends-chart');
    if (!ctx) return;
    
    if (trendsChart) {
        trendsChart.destroy();
    }
    
    trendsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Positive',
                    data: positive,
                    backgroundColor: '#10b981',
                    borderRadius: 6,
                    borderSkipped: false
                },
                {
                    label: 'Neutral',
                    data: neutral,
                    backgroundColor: '#f59e0b',
                    borderRadius: 6,
                    borderSkipped: false
                },
                {
                    label: 'Negative',
                    data: negative,
                    backgroundColor: '#ef4444',
                    borderRadius: 6,
                    borderSkipped: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 13,
                            family: 'Inter'
                        },
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        family: 'Inter'
                    },
                    bodyFont: {
                        size: 13,
                        family: 'Inter'
                    },
                    callbacks: {
                        footer: (items) => {
                            let sum = 0;
                            items.forEach(item => sum += item.parsed.y);
                            return 'Total: ' + sum;
                        }
                    }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            family: 'Inter',
                            size: 11
                        }
                    }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    grid: {
                        color: '#f3f4f6'
                    },
                    ticks: {
                        font: {
                            family: 'Inter'
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            family: 'Inter'
                        }
                    }
                }
            }
        }
    });
}

// Show Error Message
function showError(message) {
    // You can implement a toast notification here
    console.error(message);
    alert(message);
}

// Export functions for HTML onclick
window.predictSentiment = predictSentiment;
