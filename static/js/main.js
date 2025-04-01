document.addEventListener('DOMContentLoaded', () => {
    console.log("main.js cargado");

    // Restablecer estado de la galería al cargar o recargar
    const galleryItems = document.querySelectorAll('.galeria-item');
    if (galleryItems.length > 0 && typeof gsap !== 'undefined') {
        galleryItems.forEach(item => {
            gsap.set(item, { y: 0, scale: 1, boxShadow: '0 5px 15px rgba(0, 0, 0, 0.1)' });
            const cover = item.querySelector('.galeria-cover');
            const title = item.querySelector('.galeria-title');
            if (cover) gsap.set(cover, { scale: 1, opacity: 1 });
            if (title) gsap.set(title, { y: 0, color: '#fff' });
        });
    }

    // Verificar si GSAP y ScrollTrigger están disponibles
    if (typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
        gsap.registerPlugin(ScrollTrigger);
    } else {
        console.warn("GSAP o ScrollTrigger no están disponibles. Algunas animaciones no funcionarán.");
    }

    // Elementos del héroe
    const heroSubtitle = document.querySelector('.hero-subtitle');
    const heroBtn = document.querySelector('.hero-btn');
    const heroOverlay = document.querySelector('.hero-overlay');
    const navbar = document.querySelector('.navbar');

    // Animaciones del héroe
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

    // Animación inicial del navbar
    if (navbar && typeof gsap !== 'undefined') {
        gsap.from(navbar, {
            y: -100,
            duration: 1,
            ease: 'power2.out',
            onComplete: () => {
                navbar.style.willChange = 'auto';
                navbar.style.transform = 'translateZ(0)';
            }
        });
    }

    // Control de visibilidad del navbar al hacer scroll
    let lastScrollTop = 0;
    window.addEventListener('scroll', () => {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop;
        const navbarHeight = 60;

        if (navbar) {
            if (scrollTop <= navbarHeight) {
                navbar.classList.remove('hidden');
                navbar.classList.toggle('scrolled', scrollTop > 50);
            } else if (scrollTop > lastScrollTop) {
                navbar.classList.add('hidden');
            } else {
                navbar.classList.remove('hidden');
            }
            navbar.classList.toggle('scrolled', scrollTop > 50);
            lastScrollTop = scrollTop <= 0 ? 0 : scrollTop;
        }
    }, { passive: true });

    // Menú hamburguesa (actualizado y consolidado)
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');
    const body = document.body;

    if (hamburger && navLinks) {
        console.log("Hamburguesa y navLinks encontrados");
        hamburger.addEventListener('click', () => {
            console.log("Hamburguesa clicada");
            hamburger.classList.toggle('active');
            navLinks.classList.toggle('active');
            body.classList.toggle('menu-open'); // Bloquea/desbloquea el scroll
        });

        // Cerrar el menú al hacer clic fuera
        document.addEventListener('click', (e) => {
            if (!hamburger.contains(e.target) && !navLinks.contains(e.target)) {
                hamburger.classList.remove('active');
                navLinks.classList.remove('active');
                body.classList.remove('menu-open');
            }
        });

        // Cerrar el menú al hacer clic en un enlace
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navLinks.classList.remove('active');
                body.classList.remove('menu-open');
            });
        });
    } else {
        console.error("Hamburguesa o navLinks no encontrados");
    }

    // Dropdowns en el menú móvil
    const dropdowns = document.querySelectorAll('.dropdown');
    if (dropdowns.length > 0) {
        dropdowns.forEach(dropdown => {
            const submenu = dropdown.querySelector('.submenu');
            const dropdownLink = dropdown.querySelector('a');

            if (submenu && dropdownLink) {
                dropdownLink.addEventListener('click', (e) => {
                    if (window.innerWidth <= 768) {
                        e.preventDefault();
                        submenu.classList.toggle('active');
                    }
                });
            }
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

    // Animaciones de hover para la galería con GSAP
    if (typeof gsap !== 'undefined') {
        const galleryItems = document.querySelectorAll('.galeria-item');
        const isMobile = window.innerWidth < 768; // Detectar pantallas móviles

        galleryItems.forEach(item => {
            const cover = item.querySelector('.galeria-cover');
            const title = item.querySelector('.galeria-title');

            // Restablecer estado inicial al cargar o recargar la página
            gsap.set(item, { y: 0, scale: 1, boxShadow: '0 5px 15px rgba(0, 0, 0, 0.1)' });
            if (cover) gsap.set(cover, { scale: 1, opacity: 1 });
            if (title) gsap.set(title, { y: 0, color: '#fff' });

            // Solo aplicar animaciones de hover si no es móvil
            if (!isMobile) {
                // Animación al entrar con el mouse
                item.addEventListener('mouseenter', () => {
                    gsap.to(item, {
                        y: -20, // Sube 20px
                        scale: 1.05, // Escala un 5%
                        boxShadow: '0 15px 30px rgba(0, 0, 0, 0.25), 0 0 20px rgba(255, 202, 40, 0.5)', // Sombra + resplandor
                        duration: 0.4,
                        ease: 'elastic.out(1, 0.5)', // Rebote elástico
                    });
                    if (cover) {
                        gsap.to(cover, {
                            scale: 1.1, // Zoom en la imagen
                            opacity: 0.85, // Leve desvanecimiento
                            duration: 0.4,
                            ease: 'power2.out'
                        });
                    }
                    if (title) {
                        gsap.to(title, {
                            y: -10, // Sube el título 10px
                            color: '#ffca28', // Cambio de color
                            duration: 0.4,
                            ease: 'power2.out'
                        });
                    }
                });

                // Animación al salir con el mouse
                item.addEventListener('mouseleave', () => {
                    gsap.to(item, {
                        y: 0, // Vuelve a la posición original
                        scale: 1, // Escala normal
                        boxShadow: '0 5px 15px rgba(0, 0, 0, 0.1)', // Sombra original sin resplandor
                        duration: 0.4,
                        ease: 'power2.out'
                    });
                    if (cover) {
                        gsap.to(cover, {
                            scale: 1, // Escala normal
                            opacity: 1, // Opacidad completa
                            duration: 0.4,
                            ease: 'power2.out'
                        });
                    }
                    if (title) {
                        gsap.to(title, {
                            y: 0, // Vuelve a la posición original
                            color: '#fff', // Color original
                            duration: 0.4,
                            ease: 'power2.out'
                        });
                    }
                });
            }
        });
    }

    // Modo oscuro
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const toggleIcon = darkModeToggle ? darkModeToggle.querySelector('.dark-mode-icon') : null;

    if (darkModeToggle && body) {
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        if (isDarkMode) {
            body.classList.add('dark-mode');
            if (toggleIcon) toggleIcon.classList.add('light-icon');
        } else {
            if (toggleIcon) toggleIcon.classList.add('dark-icon');
        }

        darkModeToggle.addEventListener('click', () => {
            body.classList.toggle('dark-mode');
            const darkModeEnabled = body.classList.contains('dark-mode');
            localStorage.setItem('darkMode', darkModeEnabled);
            if (toggleIcon) {
                toggleIcon.classList.toggle('light-icon');
                toggleIcon.classList.toggle('dark-icon');
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