const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultContainer = document.getElementById('result-container');
const rateLimitMessage = document.getElementById('rate-limit-message');
let rateLimitEndTime = 0;
const rateLimitInterval = 15000; // 15 seconds

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
    rateLimitMessage.textContent = '';
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
        setTimeout(() => {
            dropZone.classList.remove('disabled');
            fileInput.disabled = false;
        }, rateLimitInterval);
    }
}

function displayResults(results) {
    resultContainer.innerHTML = '';
    results.forEach(result => {
        const item = document.createElement('div');
        item.className = `result-item ${result.status}`;
        item.innerHTML = `
    <img src="${result.url}" alt="${result.filename}" />
    <p>${result.filename}</p>
    <p>Score: ${result.mean_score.toFixed(2)}</p>
    <p class="${result.status}">${result.status}</p>
    `;
        resultContainer.appendChild(item);
    });
}

function updateRateLimitMessage() {
    const remainingTime = Math.ceil((rateLimitEndTime - Date.now()) / 1000);
    if (remainingTime > 0) {
        rateLimitMessage.textContent = `Please wait ${remainingTime} seconds before uploading again.`;
    } else {
        rateLimitMessage.textContent = '';
    }
}
setInterval(updateRateLimitMessage, 1000);
