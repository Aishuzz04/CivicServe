    <script>
        function toggleSidebar() {
            const sidebar = document.querySelector('aside');
            sidebar.classList.toggle('hidden');
        }

        function updateStatus(select) {
            const status = select.value;
            const badge = select;

            badge.className = 'status-badge border-0 focus:ring-2 focus:ring-blue-500';
            if (status === 'Pending') {
                badge.classList.add('bg-yellow-100', 'text-yellow-800');
            } else if (status === 'Assigned') {
                badge.classList.add('bg-blue-100', 'text-blue-800');
            } else if (status === 'Resolved') {
                badge.classList.add('bg-green-100', 'text-green-800');
            }

            alert('Status updated to: ' + status);
        }
    </script>