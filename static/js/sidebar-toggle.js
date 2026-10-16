/**
 * CivicServe - Universal Sidebar Toggle Controller
 * The sidebar will be hidden when the button is clicked,
 * and will be displayed only when the button is reclicked.
 */
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('mainContent');
    const overlay = document.getElementById('overlay');
    if (!sidebar) return;

    const isDesktop = window.innerWidth >= 1024;

    if (isDesktop) {
        // Desktop collapse / expand
        // On desktop, the sidebar is visible by default unless marked with .sidebar-hidden
        const isHidden = sidebar.classList.contains('sidebar-hidden') || sidebar.style.transform === 'translateX(-100%)';

        if (isHidden) {
            // Reclicked: Display sidebar again
            sidebar.classList.remove('sidebar-hidden');
            sidebar.style.removeProperty('transform');
            if (mainContent) {
                mainContent.classList.remove('content-expanded');
                mainContent.style.removeProperty('margin-left');
            }
        } else {
            // Clicked: Hide sidebar
            sidebar.classList.add('sidebar-hidden');
            sidebar.style.transform = 'translateX(-100%)';
            if (mainContent) {
                mainContent.classList.add('content-expanded');
                mainContent.style.marginLeft = '0px';
            }
        }
    } else {
        // Mobile drawer toggle
        const isMobileOpen = sidebar.classList.contains('sidebar-mobile-show') || sidebar.style.transform === 'translateX(0px)';

        if (isMobileOpen) {
            // Reclicked: Hide sidebar
            sidebar.classList.remove('sidebar-mobile-show');
            sidebar.style.transform = 'translateX(-100%)';
            if (overlay) overlay.classList.add('hidden');
        } else {
            // Clicked: Display sidebar
            sidebar.classList.add('sidebar-mobile-show');
            sidebar.style.transform = 'translateX(0px)';
            if (overlay) overlay.classList.remove('hidden');
        }
    }
}

// Reset inline styles on window resize across breakpoints
window.addEventListener('resize', function() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('mainContent');
    const overlay = document.getElementById('overlay');
    if (!sidebar) return;

    if (window.innerWidth >= 1024) {
        if (overlay) overlay.classList.add('hidden');
        sidebar.classList.remove('sidebar-mobile-show');
        if (sidebar.classList.contains('sidebar-hidden')) {
            sidebar.style.transform = 'translateX(-100%)';
            if (mainContent) mainContent.style.marginLeft = '0px';
        } else {
            sidebar.style.removeProperty('transform');
            if (mainContent) mainContent.style.removeProperty('margin-left');
        }
    } else {
        if (sidebar.classList.contains('sidebar-mobile-show')) {
            sidebar.style.transform = 'translateX(0px)';
            if (overlay) overlay.classList.remove('hidden');
        } else {
            sidebar.style.transform = 'translateX(-100%)';
            if (overlay) overlay.classList.add('hidden');
        }
        if (mainContent) mainContent.style.marginLeft = '0px';
    }
});
