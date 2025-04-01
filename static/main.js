document.addEventListener('DOMContentLoaded', () => {
    console.log("main.js cargado");

    // Verificar si GSAP y ScrollTrigger están disponibles
    if (typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
        gsap.registerPlugin(ScrollTrigger);
    } else {
        console.warn("GSAP o ScrollTrigger no están disponibles. Algunas animaciones no funcionarán.");
    }

    const heroSubtitle = document.querySelector('.hero-subtitle');
    const heroBtn = document.querySelector('.hero-btn');
    const heroOverlay = document.querySelector('.hero-overlay');

    if (heroSubtitle && typeof gsap !== 'undefined') {
        gsap.from(heroSubtitle, { opacity: 0, y: 30, duration: 1.5, ease: 'power3.out' });
    }
    if (heroBtn && typeof gsap !== 'undefined') {
        gsap.from(heroBtn, { opacity: 0, scale: 0.5, duration: 1.2, delay: 0.6, ease: 'elastic.out(1, 0.5)' });
    }
    if (heroOverlay && typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
        gsap.to(heroOverlay, {
            y: '20%',
            ease: 'none',
            scrollTrigger: {
                trigger: '.hero',
                start: 'top top',
                end: 'bottom top',
                scrub: true
            }
        });
    }

    // Animación inicial del Navbar y fijarlo después
    if (typeof gsap !== 'undefined') {
        gsap.from('.navbar', { 
            y: -100, 
            duration: 1, 
            ease: 'power2.out',
            onComplete: () => {
                // Fijar el navbar en su posición final después de la animación
                gsap.set('.navbar', { clearProps: 'all' }); // Limpia propiedades animadas
                const navbar = document.querySelector('.navbar');
                navbar.style.position = 'fixed';
                navbar.style.top = '0';
                navbar.style.left = '0';
                navbar.style.width = '100%';
                navbar.style.transform = 'none'; // Asegura que no haya transformaciones residuales
                navbar.classList.remove('hidden'); // Evita que se aplique la clase hidden
            }
        });
    }

    // Menú hamburguesa
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');
    if (hamburger && navLinks) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navLinks.classList.toggle('active');
            // Asegurar que el navbar no se vea afectado
            const navbar = document.querySelector('.navbar');
            navbar.style.transform = 'none'; // Forzar que el navbar no se mueva
        });

        // Cerrar el menú al hacer clic en un enlace
        const navItems = navLinks.querySelectorAll('a');
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navLinks.classList.remove('active');
                // Forzar que el navbar permanezca visible
                const navbar = document.querySelector('.navbar');
                navbar.style.transform = 'none';
                navbar.classList.remove('hidden');
            });
        });
    }

    // Buscador minimalista
    const searchToggle = document.querySelector('.search-toggle');
    const searchForm = document.querySelector('.search-form');

    if (searchToggle && searchForm) {
        searchToggle.addEventListener('click', (e) => {
            e.preventDefault();
            searchForm.classList.toggle('active');
            if (searchForm.classList.contains('active')) {
                const searchInput = document.querySelector('.search-input');
                if (searchInput) searchInput.focus();
            }
        });
    } else {
        console.error('Error: .search-toggle o .search-form no encontrados en el DOM');
    }

    // Animaciones optimizadas con Intersection Observer
    const observerOptions = {
        root: null,
        rootMargin: '200px 0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                const el = entry.target;
                if (el.dataset.aos === 'fade-up') {
                    gsap.fromTo(el, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.8, delay: index * 0.1, ease: 'power2.out', onComplete: () => observer.unobserve(el) });
                } else if (el.dataset.aos === 'zoom-in') {
                    gsap.fromTo(el, { opacity: 0, scale: 0.95 }, { opacity: 1, scale: 1, duration: 0.6, delay: index * 0.05, ease: 'power2.out', onComplete: () => observer.unobserve(el) });
                } else if (el.dataset.aos === 'fade-down') {
                    gsap.fromTo(el, { opacity: 0, y: -30 }, { opacity: 1, y: 0, duration: 0.8, delay: index * 0.1, ease: 'power2.out', onComplete: () => observer.unobserve(el) });
                }
            }
        });
    }, observerOptions);

    document.querySelectorAll('[data-aos]').forEach(el => observer.observe(el));

    // Modo oscuro
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const body = document.body;
    const toggleIcon = darkModeToggle ? darkModeToggle.querySelector('.toggle-icon') : null;

    if (darkModeToggle && body) {
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        if (isDarkMode) {
            body.classList.add('dark-mode');
            if (toggleIcon) toggleIcon.textContent = '☀️';
        } else {
            if (toggleIcon) toggleIcon.textContent = '🌙';
        }

        darkModeToggle.addEventListener('click', () => {
            body.classList.toggle('dark-mode');
            const darkModeEnabled = body.classList.contains('dark-mode');
            localStorage.setItem('darkMode', darkModeEnabled);
            if (toggleIcon) {
                toggleIcon.textContent = darkModeEnabled ? '☀️' : '🌙';
            }
            gsap.to(body, {
                backgroundColor: darkModeEnabled ? '#1a1a1a' : '#f8f1e9',
                color: darkModeEnabled ? '#e0e0e0' : '#333',
                duration: 0.5,
                ease: 'power2.out'
            });
        });
    } else {
        console.error('Error: #dark-mode-toggle o body no encontrados en el DOM');
    }

    // Mostrar y ocultar mensajes flash
    const flashMessages = document.querySelectorAll('.flash');
    flashMessages.forEach(message => {
        message.classList.add('show');
        setTimeout(() => {
            message.classList.remove('show');
        }, 3000);
    });

    // Búsqueda avanzada con Fuse.js (solo en la página de búsqueda)
    if (document.querySelector('.busqueda')) {
        const reflexiones = window.reflexionesData || [];
        const fuse = new Fuse(reflexiones, {
            keys: ['titulo', 'contenido', 'categoria'],
            threshold: 0.3,
            includeScore: true
        });

        const searchInput = document.querySelector('.search-input');
        const resultsList = document.getElementById('results-list');
        const totalResults = document.getElementById('total-results');
        const searchQuery = document.getElementById('search-query');

        if (searchInput && resultsList && totalResults && searchQuery) {
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.trim();
                if (!query) {
                    resultsList.innerHTML = reflexiones.map(result => `
                        <li class="resultado-item" style="border: 1px solid red; margin: 10px 0; padding: 10px;">
                            <div class="resultado-imagen">
                                ${result.imagen ? `<img src="${result.imagen}" alt="${result.titulo}" class="resultado-cover" style="border: 2px solid green; max-width: 200px; height: auto;">` : '<p>Sin imagen</p>'}
                            </div>
                            <div class="resultado-content">
                                <h2><a href="/reflexion/${result.id}">${result.titulo}</a></h2>
                                <p class="categoria">${result.categoria.charAt(0).toUpperCase() + result.categoria.slice(1)}</p>
                                <div class="resultado-extracto">
                                    <p>${result.contenido.substring(0, 200)}${result.contenido.length > 200 ? '...' : ''}</p>
                                </div>
                            </div>
                        </li>
                    `).join('');
                    totalResults.textContent = reflexiones.length;
                    searchQuery.textContent = '';
                    return;
                }

                const results = fuse.search(query);
                totalResults.textContent = results.length;
                searchQuery.textContent = query;

                resultsList.innerHTML = results.map(result => `
                    <li class="resultado-item" style="border: 1px solid red; margin: 10px 0; padding: 10px;">
                        <div class="resultado-imagen">
                            ${result.item.imagen ? `<img src="${result.item.imagen}" alt="${result.item.titulo}" class="resultado-cover" style="border: 2px solid green; max-width: 200px; height: auto;">` : '<p>Sin imagen</p>'}
                        </div>
                        <div class="resultado-content">
                            <h2><a href="/reflexion/${result.item.id}">${result.item.titulo}</a></h2>
                            <p class="categoria">${result.item.categoria.charAt(0).toUpperCase() + result.item.categoria.slice(1)}</p>
                            <div class="resultado-extracto">
                                <p>${result.item.contenido.substring(0, 200)}${result.item.contenido.length > 200 ? '...' : ''}</p>
                            </div>
                        </div>
                    </li>
                `).join('') || `<p class="no-resultados">No se encontraron resultados para "${query}".</p>`;
            });
        }
    }

    // Convertir imágenes inline para lazy loading
    const inlineImages = document.querySelectorAll('.blog-text img[src], .post-content img[src]');
    inlineImages.forEach(img => {
        if (!img.classList.contains('lazyload')) {
            const src = img.getAttribute('src');
            img.setAttribute('data-src', src);
            img.removeAttribute('src');
            img.classList.add('lazyload');
        }
    });
});