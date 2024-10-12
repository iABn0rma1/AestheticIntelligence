const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultContainer = document.getElementById('result-container');
const rateLimitMessage = document.getElementById('rate-limit-message');
const spinner = document.getElementById('spinner'); // Get the spinner element
let rateLimitEndTime = 0;
const rateLimitInterval = 69000; // 69 seconds
const maxFileSize = 16 * 1024 * 1024; // 2 MB in bytes

// Event listeners for DnD
dropZone.addEventListener('click', () => {
    if (!dropZone.classList.contains('disabled')) {
        fileInput.click();
    }
});
dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!dropZone.classList.contains('disabled')) {
        dropZone.style.backgroundColor = '#333';
    }
});
dropZone.addEventListener('dragleave', () => {
    dropZone.style.backgroundColor = '#1e1e1e';
});
dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.style.backgroundColor = '#1e1e1e';
    if (!dropZone.classList.contains('disabled')) {
        handleFiles(event.dataTransfer.files);
    }
});

// file handlers
fileInput.addEventListener('change', () => {
    if (!dropZone.classList.contains('disabled')) {
        handleFiles(fileInput.files);
    }
});

async function handleFiles(files) {
    const currentTime = Date.now();
    if (currentTime < rateLimitEndTime) {
        const remainingTime = Math.ceil((rateLimitEndTime - currentTime) / 1000);
        rateLimitMessage.textContent = `Please wait ${remainingTime} seconds before uploading again.`;
        return;
    }

    // Check file size before proceeding
    const oversizedFiles = Array.from(files).filter(file => file.size > maxFileSize);
    if (oversizedFiles.length > 0) {
        rateLimitMessage.textContent = `File size exceeds the 2MB limit: ${oversizedFiles.map(f => f.name).join(', ')}.`;
        return; // Stop execution here if file is oversized
    }

    rateLimitMessage.textContent = ''; // Clear message only if files are valid
    spinner.style.display = 'block'; // Show spinner when upload starts

    dropZone.classList.add('disabled');
    fileInput.disabled = true;

    const formData = new FormData();
    Array.from(files).forEach(file => formData.append('files', file));

    try {
        const response = await fetch('/curate/', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        displayResults(result);
        rateLimitEndTime = Date.now() + rateLimitInterval;
        updateRateLimitMessage();
    } catch (error) {
        console.error('Error:', error);
    } finally {
        spinner.style.display = 'none'; // Hide spinner once upload completes
        setTimeout(() => {
            dropZone.classList.remove('disabled');
            fileInput.disabled = false;
            // Clear the rate limit message when the time is up
            rateLimitMessage.textContent = '';
        }, rateLimitInterval);
    }
}

function updateRateLimitMessage() {
    const remainingTime = Math.ceil((rateLimitEndTime - Date.now()) / 1000);
    if (remainingTime > 1) {
        rateLimitMessage.textContent = `Please wait ${remainingTime} seconds before uploading again.`;
    } else {
        rateLimitMessage.textContent = ''; // Clear message when time is up
    }
}

setInterval(updateRateLimitMessage, 1000);

function displayResults(results) {
    resultContainer.innerHTML = '';
    results.forEach(result => {
        const item = document.createElement('div');
        item.className = `result-item ${result.status}`;
        item.innerHTML = `
            <img class="lazyload" src="https://via.placeholder.com/150x75.png?text=Loading" data-src="${result.url}" alt="${result.filename}" />
            <p>${result.filename}</p>
            <p>Score: ${result.mean_score.toFixed(2)}</p>
            <p class="${result.status}">${result.status}</p>
        `;
        resultContainer.appendChild(item);
    });

    // Initialize lazy loading for the newly added images
    lazyLoadImages();
}

function lazyLoadImages() {
    let lazyImages = [].slice.call(document.querySelectorAll("img.lazyload"));

    if ("IntersectionObserver" in window) {
        let lazyImageObserver = new IntersectionObserver(function (entries, observer) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    let lazyImage = entry.target;
                    lazyImage.src = lazyImage.dataset.src;
                    lazyImage.classList.remove("lazyload");
                    lazyImageObserver.unobserve(lazyImage);
                }
            });
        });

        lazyImages.forEach(function (lazyImage) {
            lazyImageObserver.observe(lazyImage);
        });
    } else {
        // Fallback for browsers without IntersectionObserver support
        lazyImages.forEach(function (lazyImage) {
            lazyImage.src = lazyImage.dataset.src;
        });
    }
}
