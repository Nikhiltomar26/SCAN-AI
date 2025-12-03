// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeImageBtn = document.getElementById('removeImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const results = document.getElementById('results');

// Result elements
const explanation = document.getElementById('explanation');
const highlights = document.getElementById('highlights');
const rawText = document.getElementById('rawText');

// Toggle buttons
const toggleExplanation = document.getElementById('toggleExplanation');
const toggleHighlights = document.getElementById('toggleHighlights');
const toggleRawText = document.getElementById('toggleRawText');

// Store uploaded file
let uploadedFile = null;

// Initialize event listeners
function init() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Remove image
    removeImageBtn.addEventListener('click', removeImage);
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeReport);
    
    // Toggle buttons
    toggleExplanation.addEventListener('click', () => toggleCard('explanationContent', 'toggleExplanation'));
    toggleHighlights.addEventListener('click', () => toggleCard('highlightsContent', 'toggleHighlights'));
    toggleRawText.addEventListener('click', () => toggleCard('rawTextContent', 'toggleRawText'));
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragging');
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragging');
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragging');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        processFile(file);
    }
}

// Process selected file
function processFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPEG, PNG, BMP, or TIFF)');
        return;
    }
    
    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    // Store file
    uploadedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

// Remove uploaded image
function removeImage(e) {
    e.stopPropagation();
    uploadedFile = null;
    previewImage.src = '';
    imagePreview.style.display = 'none';
    uploadArea.style.display = 'flex';
    analyzeBtn.disabled = true;
    fileInput.value = '';
    hideError();
    hideResults();
}

// Analyze report
async function analyzeReport() {
    if (!uploadedFile) return;
    
    // Show loading
    loading.style.display = 'block';
    hideError();
    hideResults();
    analyzeBtn.disabled = true;
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        // Send request
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to analyze report');
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            throw new Error('Analysis failed');
        }
        
    } catch (err) {
        showError(`Error: ${err.message}`);
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    // Populate explanation
    explanation.textContent = data.explanation;
    
    // Populate highlights
    highlights.innerHTML = '';
    data.highlights.forEach(highlight => {
        const li = document.createElement('li');
        li.textContent = highlight;
        highlights.appendChild(li);
    });
    
    // Populate raw text
    rawText.textContent = data.raw_text || 'No text extracted';
    
    // Show results
    results.style.display = 'block';
    
    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Toggle card content
function toggleCard(contentId, buttonId) {
    const content = document.getElementById(contentId);
    const button = document.getElementById(buttonId);
    
    content.classList.toggle('collapsed');
    button.classList.toggle('rotated');
}

// Show error message
function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
}

// Hide error message
function hideError() {
    error.style.display = 'none';
}

// Hide results
function hideResults() {
    results.style.display = 'none';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);