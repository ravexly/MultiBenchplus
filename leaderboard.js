// ============================================================
// MultiBench++ Leaderboard - Data, Rendering, Navigation
// ============================================================

(function () {
    'use strict';

    const METHODS = [
        'Concat', 'TF', 'Concat Early', 'LFT', 'EFT',
        'Multi-to-One', 'One-to-Multi', 'CAF', 'CACF', 'LS', 'TMC'
    ];

    const TABLES = [
        {
            id: 'table-remote-sensing',
            title: 'Table 2: Performance on Remote Sensing Datasets',
            note: '',
            mseRows: [],
            rows: [
                { name: 'Houston2013', values: [75.68, 76.53, 75.95, 60.54, 24.68, 78.54, 79.22, 74.92, 83.32, 79.43, 79.12] },
                { name: 'Houston2018', values: [70.99, 74.46, 70.07, 63.41, 35.00, 76.52, 70.89, 68.71, 77.66, 80.52, 77.33] },
                { name: 'MUUFL Gulfport', values: [84.21, 83.90, 86.26, 71.77, 46.68, 80.95, 81.23, 83.99, 86.42, 86.64, 83.20] },
                { name: 'Trento', values: [98.43, 96.95, 98.14, 95.68, 71.64, 97.71, 96.08, 97.74, 98.68, 98.53, 97.67] },
                { name: 'Berlin', values: [68.25, 70.22, 72.53, 61.72, 60.74, 73.75, 71.22, 76.31, 77.42, 78.61, 77.67] },
                { name: 'Augsburg', values: [89.24, 85.86, 89.50, 82.88, 57.29, 85.86, 86.80, 87.92, 89.05, 89.49, 87.21] },
                { name: 'ForestNet', values: [45.18, 45.63, 45.68, 47.19, 44.33, 45.78, 45.58, 45.03, 45.93, 46.08, 45.68] },
            ]
        },
        {
            id: 'table-medical-ai',
            title: 'Table 3: Performance on Medical AI Datasets',
            note: 'The dash <code>-</code> indicates the method is not applicable to this dataset.',
            mseRows: [],
            rows: [
                { name: 'TCGA-BRCA', values: [78.17, 77.18, 77.18, 69.44, 65.67, 76.79, 76.19, 78.77, 78.37, 77.18, 75.40] },
                { name: 'ROSMAP', values: [70.95, 75.71, 70.00, 42.86, 69.52, 69.52, 68.10, 66.67, 71.90, 71.43, 66.67] },
                { name: 'SIIM-ISIC', values: [97.86, 97.77, 97.86, 97.83, 97.85, 97.85, 97.83, 97.86, 97.83, 97.83, 97.85] },
                { name: 'Derm7pt', values: [45.49, 52.72, 45.41, 43.96, 38.86, 45.92, 46.77, 48.13, 52.72, 46.34, 52.72] },
                { name: 'GAMMA', values: [61.43, 62.86, 63.81, 62.38, 59.05, 57.62, 65.71, 63.33, 63.33, 62.38, 61.43] },
                { name: 'MIMIC-III', values: [68.09, 68.81, 68.48, 68.42, 68.59, 69.01, 68.76, 68.51, 68.88, 68.47, 68.87] },
                { name: 'eICU', values: [90.05, 90.03, 90.05, 90.05, 90.05, 90.05, 90.05, 90.07, 90.05, 90.05, 90.03] },
                { name: 'TCGA', values: [51.91, '-', 53.55, 60.66, 51.37, 60.11, 53.01, 56.28, 61.75, 54.10, 62.84] },
                { name: 'MIMIC-CXR (macro-F1)', values: [0.6861, 0.8458, 0.6788, 0.1551, 0.1829, 0.5229, 0.4031, 0.6136, 0.7523, 0.7644, '-'] },
            ]
        },
        {
            id: 'table-affective',
            title: 'Table 4: Performance on Affective Computing & Social Media Understanding Datasets',
            note: 'The dash <code>-</code> indicates the method is not applicable to this dataset. MSE metrics: <strong>lower is better</strong>.',
            mseRows: ['CH-SIMS (MSE)', 'CH-SIMS v2 (MSE)'],
            rows: [
                { name: 'MELD', values: [61.34, 65.66, 62.73, 47.77, 57.92, 66.37, 64.02, 65.54, 62.22, 61.60, 65.56] },
                { name: 'IEMOCAP', values: [54.96, 54.59, 54.82, 32.64, 48.14, 54.49, 54.16, 54.26, 55.06, 54.30, 51.22] },
                { name: 'MAMI', values: [70.00, 66.47, 66.13, 67.87, 66.63, 68.97, 64.50, 65.40, 67.23, 67.80, 69.73] },
                { name: 'Memotion', values: [77.89, 77.75, 78.14, 78.09, 78.09, 78.09, 78.09, 78.14, 78.09, 78.19, 78.09] },
                { name: 'MUTE', values: [67.87, 67.31, 66.59, 65.63, 68.19, 66.03, 66.99, 65.87, 67.07, 67.95, 68.67] },
                { name: 'MultiOFF', values: [56.95, 59.86, 55.38, 57.76, 59.11, 54.38, 59.33, 59.76, 62.03, 54.16, 60.01] },
                { name: 'MET-Meme(C)', values: [35.67, 34.87, 36.92, 29.68, 24.12, 33.55, 35.09, 32.89, 36.62, 34.94, 33.85] },
                { name: 'MET-Meme(E)', values: [42.95, 42.47, 41.83, 32.69, 29.81, 44.23, 39.74, 40.38, 41.03, 43.11, 42.47] },
                { name: 'Twitter2015', values: [76.05, 76.37, 75.76, 63.39, 68.05, 74.12, 70.40, 75.31, 75.70, 75.86, 64.45] },
                { name: 'Twitter1517', values: [76.83, 76.72, 76.68, 76.76, 76.68, 76.29, 76.72, 76.54, 75.97, 76.15, 76.86] },
                { name: 'CH-SIMS (MSE)', values: [0.4835, 0.7431, 0.4790, 0.4775, 0.4804, 0.4772, 0.4836, 0.4794, 0.4824, 0.4898, '-'] },
                { name: 'CH-SIMS v2 (MSE)', values: [0.3202, 0.3566, 0.3338, 0.3214, 0.3391, 0.3351, 0.3448, 0.2801, 0.3361, 0.3360, '-'] },
            ]
        },
        {
            id: 'table-other',
            title: 'Table 5: Performance on Other Datasets',
            note: '',
            mseRows: [],
            rows: [
                { name: 'MIRFLICKR', values: [62.81, 62.33, 62.42, 50.25, 38.02, 58.44, 59.95, 59.13, 62.51, 62.45, 62.00] },
                { name: 'CUB Image-Caption', values: [77.90, 76.26, 78.04, 2.89, 1.81, 71.90, 69.19, 21.09, 73.95, 79.48, 77.73] },
                { name: 'SUN-RGBD', values: [60.28, 59.43, 60.83, 45.22, 31.61, 53.52, 56.97, 53.53, 58.14, 60.78, 59.00] },
                { name: 'NYUDv2', values: [59.02, 61.47, 58.82, 47.96, 31.70, 59.58, 63.20, 60.55, 64.02, 66.87, 66.16] },
                { name: 'UPMC-Food101', values: [91.95, 92.04, 91.80, 83.76, 8.75, 86.04, 88.80, 86.89, 90.12, 91.66, 92.02] },
                { name: 'MVSA-Single', values: [79.83, 78.36, 78.29, 68.40, 63.39, 79.32, 77.78, 79.51, 78.03, 79.25, 67.50] },
                { name: 'MNIST-SVHN', values: [96.41, 96.64, 96.42, 62.11, 93.06, 95.45, 93.47, 95.35, 96.45, 96.46, 96.95] },
                { name: 'N-MNIST+N-TIDIGITS', values: [94.99, 94.28, 95.26, 80.99, 30.45, 93.52, 94.34, 94.06, 95.26, 94.88, 94.23] },
                { name: 'E-MNIST+EEG', values: [58.72, 58.21, 61.28, 17.69, 7.95, 42.56, 30.51, 49.74, 59.49, 62.05, 57.69] },
            ]
        }
    ];

    function highlightBestValues(tableData, tbody, rows) {
        const trs = tbody.querySelectorAll('tr');

        trs.forEach((tr, rowIdx) => {
            const rowData = rows[rowIdx];
            const isMSE = tableData.mseRows.includes(rowData.name);
            const valueTds = Array.from(tr.querySelectorAll('td')).slice(1);

            const numericEntries = [];
            valueTds.forEach((td, i) => {
                const val = rowData.values[i];
                if (typeof val === 'number') {
                    numericEntries.push({ td, val });
                }
            });

            if (numericEntries.length === 0) return;

            const bestVal = isMSE
                ? Math.min(...numericEntries.map(e => e.val))
                : Math.max(...numericEntries.map(e => e.val));

            numericEntries.forEach(entry => {
                if (entry.val === bestVal) {
                    entry.td.classList.add('best-cell');
                }
            });
        });
    }

    function renderTable(tableData) {
        const card = document.createElement('div');
        card.className = 'leaderboard-card';
        card.id = tableData.id;

        const titleBlock = document.createElement('div');
        titleBlock.className = 'table-title-block';
        titleBlock.innerHTML = `
            <div class="table-title-left">
                <div class="table-title-bar"></div>
                <div class="table-title-text">
                    <h3>${tableData.title}</h3>
                    ${tableData.note ? `<p class="table-note">${tableData.note}</p>` : ''}
                </div>
            </div>
        `;
        card.appendChild(titleBlock);
        const wrapper = document.createElement('div');
        wrapper.className = 'table-scroll-wrapper';

        const table = document.createElement('table');
        table.className = 'leaderboard-table';

        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');

        const datasetTh = document.createElement('th');
        datasetTh.textContent = 'Dataset';
        headerRow.appendChild(datasetTh);

        METHODS.forEach(method => {
            const th = document.createElement('th');
            th.textContent = method;
            headerRow.appendChild(th);
        });

        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        tableData.rows.forEach(row => {
            const tr = document.createElement('tr');
            const tdName = document.createElement('td');
            tdName.textContent = row.name;
            tr.appendChild(tdName);

            row.values.forEach(val => {
                const td = document.createElement('td');
                if (val === '-') {
                    td.textContent = '-';
                    td.classList.add('na-cell');
                } else {
                    td.textContent = val;
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });

        table.appendChild(tbody);
        highlightBestValues(tableData, tbody, tableData.rows);
        wrapper.appendChild(table);
        card.appendChild(wrapper);

        return card;
    }

    function buildLeaderboard() {
        const container = document.getElementById('leaderboard-tables');
        if (!container) return;
        TABLES.forEach(tableData => {
            container.appendChild(renderTable(tableData));
        });
    }

    function setupSidebarTracking() {
        const sidebarLinks = document.querySelectorAll('.sidebar-nav a');
        if (sidebarLinks.length === 0) return;

        const sections = [];
        sidebarLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                const el = document.getElementById(href.slice(1));
                if (el) sections.push({ el, link });
            }
        });

        if (sections.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                const matchedSection = sections.find(s => s.el === entry.target);
                if (!matchedSection || !entry.isIntersecting) return;
                sidebarLinks.forEach(l => l.classList.remove('active'));
                matchedSection.link.classList.add('active');
            });
        }, {
            rootMargin: '-80px 0px -60% 0px',
            threshold: 0
        });

        sections.forEach(s => observer.observe(s.el));

        sidebarLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const href = link.getAttribute('href');
                if (!href || !href.startsWith('#')) return;
                e.preventDefault();
                const target = document.getElementById(href.slice(1));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        buildLeaderboard();
        setupSidebarTracking();
    });
})();
