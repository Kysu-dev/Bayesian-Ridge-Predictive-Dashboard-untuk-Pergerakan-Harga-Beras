// Setup Chart.js defaults
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
Chart.defaults.color = '#6b7280';

// Register annotation plugin if available
if (typeof ChartAnnotation !== 'undefined') {
    Chart.register(ChartAnnotation);
}

// Function to render main dashboard chart
function renderDashboardChart(canvasId, timeRange = 'all') {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    // Filter data based on time range
    let filteredActual = [...ACTUAL_DATA];
    if (timeRange === '1y') {
        filteredActual = filteredActual.slice(-12);
    } else if (timeRange === '2y') {
        filteredActual = filteredActual.slice(-24);
    }

    const labels = [...filteredActual.map(d => d.month), ...FORECAST_DATA.map(d => d.month)];
    const actualPrices = [...filteredActual.map(d => d.price), ...FORECAST_DATA.map(() => null)];
    
    // Connect the line visually
    const forecastPrices = Array(filteredActual.length).fill(null);
    forecastPrices[filteredActual.length - 1] = filteredActual[filteredActual.length - 1].price;
    FORECAST_DATA.forEach(d => forecastPrices.push(d.price));

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Harga Aktual',
                    data: actualPrices,
                    borderColor: '#2d8c4e', // Primary green
                    backgroundColor: 'rgba(45, 140, 78, 0.1)',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    fill: true,
                    tension: 0.2
                },
                {
                    label: 'Prediksi Model',
                    data: forecastPrices,
                    borderColor: '#EF9F27', // Accent amber
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointBackgroundColor: '#EF9F27',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: false,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR' }).format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            xMin: filteredActual.length - 1,
                            xMax: filteredActual.length - 1,
                            borderColor: '#9ca3af',
                            borderWidth: 1,
                            borderDash: [2, 2],
                            label: {
                                display: true,
                                content: 'Hari Ini (Apr 2026)',
                                position: 'start'
                            }
                        }
                    }
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function(value) {
                            return 'Rp ' + value.toLocaleString('id-ID');
                        }
                    }
                }
            }
        }
    });
}

function renderHistorisChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: ACTUAL_DATA.map(d => d.month),
            datasets: [{
                label: 'Harga Aktual Nasional',
                data: ACTUAL_DATA.map(d => d.price),
                borderColor: '#2d8c4e',
                backgroundColor: 'rgba(45, 140, 78, 0.1)',
                borderWidth: 2,
                pointRadius: 1,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function renderModelChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const recentData = ACTUAL_DATA.slice(-12);
    // Mock predicted values slightly offset from actual
    const predictedData = recentData.map(d => d.price + (Math.random() * 40 - 20));

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: recentData.map(d => d.month),
            datasets: [
                {
                    label: 'Aktual',
                    data: recentData.map(d => d.price),
                    borderColor: '#2d8c4e',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: 'Prediksi (Fitted)',
                    data: predictedData,
                    borderColor: '#EF9F27',
                    borderWidth: 2,
                    borderDash: [4, 4],
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function renderFeatureImportance(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Lag-1 (Harga Kemarin)', 'Moving Avg 3M', 'Lag-2', 'Bulan (Musiman)'],
            datasets: [{
                label: 'Importance',
                data: [0.72, 0.58, 0.41, 0.19],
                backgroundColor: '#2d8c4e',
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function renderPrediksiSmallChart(canvasId, resultData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    // Destroy existing chart if any
    let chartStatus = Chart.getChart(canvasId);
    if (chartStatus != undefined) {
      chartStatus.destroy();
    }

    const recentData = ACTUAL_DATA.slice(-6);
    const labels = [...recentData.map(d => d.month), resultData.month];
    const actual = [...recentData.map(d => d.price), null];
    const forecast = [...recentData.map(() => null)];
    forecast[forecast.length - 1] = recentData[recentData.length - 1].price;
    forecast.push(resultData.predicted);

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Aktual',
                    data: actual,
                    borderColor: '#2d8c4e',
                    tension: 0.2
                },
                {
                    label: 'Prediksi',
                    data: forecast,
                    borderColor: '#EF9F27',
                    borderDash: [5, 5],
                    pointBackgroundColor: '#EF9F27',
                    pointRadius: 5,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });
}
