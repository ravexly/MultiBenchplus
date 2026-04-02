(function () {
    'use strict';

    function updateHeaderState(header) {
        if (!header) return;
        header.classList.toggle('scrolled', window.scrollY > 50);
    }

    function setupMobileMenu(menuButton, navLinks) {
        if (!menuButton || !navLinks) return;

        const closeMenu = () => {
            navLinks.classList.remove('active');
            menuButton.textContent = 'Menu';
        };

        menuButton.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            menuButton.textContent = navLinks.classList.contains('active') ? 'Close' : 'Menu';
        });

        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', closeMenu);
        });
    }

    function copyCitation() {
        const citation = document.getElementById('bibtex');
        const button = document.getElementById('copy-citation-btn');
        if (!citation || !button) return;

        navigator.clipboard.writeText(citation.innerText).then(() => {
            const originalText = button.textContent;
            button.textContent = 'Copied';
            button.classList.add('success');

            setTimeout(() => {
                button.textContent = originalText;
                button.classList.remove('success');
            }, 1800);
        }).catch(error => {
            console.error('Failed to copy citation text:', error);
        });
    }

    function setupCitationCopy() {
        const button = document.getElementById('copy-citation-btn');
        if (!button) return;
        button.addEventListener('click', copyCitation);
    }

    function setupDatasetAccordion() {
        const root = document.getElementById('dataset-accordion-root');
        if (!root) return;

        root.addEventListener('click', (event) => {
            const header = event.target.closest('.accordion-header');
            if (!header || !root.contains(header)) return;

            const item = header.closest('.accordion-item');
            if (!item) return;

            const isActive = item.classList.toggle('active');
            header.setAttribute('aria-expanded', String(isActive));
        });

        // Keep all sections collapsed by default.
    }

    function initPage() {
        setupDatasetAccordion();
        setupCitationCopy();

        const header = document.querySelector('.site-header');
        updateHeaderState(header);
        window.addEventListener('scroll', () => updateHeaderState(header));

        const mobileMenuButton = document.querySelector('.mobile-menu-btn');
        const navLinks = document.querySelector('.nav-links');
        setupMobileMenu(mobileMenuButton, navLinks);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initPage);
    } else {
        initPage();
    }
})();
