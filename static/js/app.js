document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
            this.classList.add('active');
            
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Load Dashboard Data
    fetchDashboardData();

    // Form submission handling
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', handlePrediction);
    }
});

// Setup Charts with Chart.js
let scoreChartInstance = null;
let gradeChartInstance = null;

async function fetchDashboardData() {
    try {
        const response = await fetch('/api/dashboard-data');
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        
        // Update stats
        if (data.stats) {
            document.getElementById('stat-pass-rate').textContent = `${data.stats.pass_rate}%`;
            document.getElementById('stat-avg-score').textContent = data.stats.avg_score;
            document.getElementById('stat-study-hrs').textContent = `${data.stats.avg_study_hrs} h/d`;
            document.getElementById('stat-attendance').textContent = `${data.stats.avg_attend}%`;
            
            // Render charts
            renderScoreChart(data.stats.score_bins);
            renderGradeChart(data.stats.grade_dist);
        }
        
        // Update metrics table
        if (data.metrics) {
            renderMetricsTable(data.metrics);
        }
    } catch (error) {
        console.error('Error fetching dashboard data:', error);
    }
}

function renderScoreChart(scoreData) {
    const ctx = document.getElementById('scoreDistChart').getContext('2d');
    
    if (scoreChartInstance) scoreChartInstance.destroy();
    
    scoreChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: scoreData.labels,
            datasets: [{
                label: 'Number of Students',
                data: scoreData.counts,
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#cbd5e1' } },
                x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
            }
        }
    });
}

function renderGradeChart(gradeData) {
    const ctx = document.getElementById('gradeDistChart').getContext('2d');
    const grades = ['A', 'B', 'C', 'D', 'F'];
    const counts = grades.map(g => gradeData[g] || 0);
    
    const colors = [
        'rgba(16, 185, 129, 0.7)', // A
        'rgba(59, 130, 246, 0.7)', // B
        'rgba(139, 92, 246, 0.7)', // C
        'rgba(245, 158, 11, 0.7)', // D
        'rgba(239, 68, 68, 0.7)'   // F
    ];
    
    if (gradeChartInstance) gradeChartInstance.destroy();
    
    gradeChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: grades,
            datasets: [{
                data: counts,
                backgroundColor: colors,
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: { position: 'right', labels: { color: '#cbd5e1' } }
            }
        }
    });
}

function renderMetricsTable(metrics) {
    const tbody = document.querySelector('#metrics-table tbody');
    tbody.innerHTML = '';
    
    // Helper to format rows
    const addRow = (model, task, metric, score) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${model}</td>
            <td><span class="tech-tag" style="padding: 2px 8px; font-size: 0.8rem;">${task}</span></td>
            <td>${metric}</td>
            <td style="font-weight: 600; color: #fff;">${score}</td>
        `;
        tbody.appendChild(tr);
    };
    
    // Add best models
    if (metrics.regression_results && metrics.regression_results["Random Forest Regressor"]) {
        addRow("Random Forest", "Regression", "R² Score", metrics.regression_results["Random Forest Regressor"].R2);
    }
    if (metrics.classification_results && metrics.classification_results["Random Forest Classifier"]) {
         addRow("Random Forest", "Classification", "Accuracy", metrics.classification_results["Random Forest Classifier"].Accuracy);
    }
    if (metrics.binary_results && metrics.binary_results["Accuracy"]) {
         addRow("Random Forest (Pass/Fail)", "Binary Classification", "Accuracy", metrics.binary_results["Accuracy"]);
    }
}

async function handlePrediction(e) {
    e.preventDefault();
    
    // UI elements
    const placeholder = document.getElementById('result-placeholder');
    const content = document.getElementById('result-content');
    const loading = document.getElementById('loading-overlay');
    
    // Show loading
    placeholder.classList.add('hidden');
    content.classList.add('hidden');
    loading.classList.remove('hidden');
    
    // Gather form data
    const formData = new FormData(e.target);
    const data = {};
    for (let [key, value] of formData.entries()) {
        // Convert to appropriate types
        if (['age', 'assignments_done', 'library_visits'].includes(key)) {
            data[key] = parseInt(value);
        } else if (['previous_score', 'attendance_pct', 'study_hours_per_day', 'sleep_hours'].includes(key)) {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    }
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) throw new Error('Prediction failed');
        
        const result = await response.json();
        
        // Update UI
        updateResultUI(result);
        
        // Hide loading, show content
        setTimeout(() => {
            loading.classList.add('hidden');
            content.classList.remove('hidden');
        }, 800); // Slight delay for effect
        
    } catch (error) {
        console.error('Error during prediction:', error);
        alert('An error occurred during prediction. Please try again.');
        loading.classList.add('hidden');
        placeholder.classList.remove('hidden');
    }
}

function updateResultUI(result) {
    // Update Score
    document.getElementById('res-score').textContent = result.predicted_score;
    const scoreDeg = (result.predicted_score / 100) * 360;
    document.getElementById('score-circle').style.setProperty('--score-deg', `${scoreDeg}deg`);
    
    // Update Grade
    document.getElementById('res-grade').textContent = result.predicted_grade;
    
    // Update Pass/Fail Status
    const statusBadge = document.getElementById('res-status');
    statusBadge.textContent = result.pass_fail;
    statusBadge.className = 'status-badge ' + (result.pass_fail === 'Pass' ? 'status-pass' : 'status-fail');
    
    // Update Insights
    const insightsList = document.getElementById('res-insights');
    insightsList.innerHTML = '';
    result.insights.forEach(insight => {
        const li = document.createElement('li');
        li.textContent = insight;
        insightsList.appendChild(li);
    });
}
