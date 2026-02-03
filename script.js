// Header scroll effect
const header = document.querySelector('header');
window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
});

// Mobile menu toggle
const mobileBtn = document.querySelector('.mobile-menu-btn');
const navLinks = document.querySelector('.nav-links');

if (mobileBtn) {
    mobileBtn.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        const icon = mobileBtn.querySelector('i');
        if (navLinks.classList.contains('active')) {
            icon.classList.remove('fa-bars');
            icon.classList.add('fa-times');
        } else {
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        }
    });
}

// Close mobile menu when clicking a link
document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', () => {
        navLinks.classList.remove('active');
        if (mobileBtn) {
            const icon = mobileBtn.querySelector('i');
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        }
    });
});

// Copy citation
function copyCitation() {
    const citationText = document.getElementById('bibtex').innerText;
    navigator.clipboard.writeText(citationText).then(() => {
        const btn = document.querySelector('.btn-copy');
        const originalText = btn.innerHTML;

        btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        btn.classList.add('success');

        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.classList.remove('success');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

// Accordion functionality for dataset categories
const accordionHeaders = document.querySelectorAll('.accordion-header');

accordionHeaders.forEach(header => {
    header.addEventListener('click', () => {
        const accordionItem = header.parentElement;
        const isActive = accordionItem.classList.contains('active');

        // Optional: Close other accordions (remove this block for multiple open)
        // document.querySelectorAll('.accordion-item').forEach(item => {
        //     item.classList.remove('active');
        // });

        // Toggle current accordion
        if (isActive) {
            accordionItem.classList.remove('active');
        } else {
            accordionItem.classList.add('active');
        }
    });
});

// Open the first accordion by default
const firstAccordion = document.querySelector('.accordion-item');
if (firstAccordion) {
    firstAccordion.classList.add('active');
}
