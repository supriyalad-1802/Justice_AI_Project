// script.js

// --- Global Elements ---
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const summarizeBtn = document.getElementById('summarizeBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const fileNameElem = document.getElementById('selectedFileName');

// --- Helper Functions ---
function updateFileDisplay(file) {
    fileNameElem.textContent = `Selected file: ${file.name}`;
}

// --- Upload Page Logic ---
// This logic will only apply when window.location.pathname is '/upload'
if (window.location.pathname === '/upload') {
    // Event Listeners for file selection and drag & drop
    dropZone.addEventListener('click', (e) => {
        // Ensure clicking the dropZone (but not the file input itself if it's visible) triggers the file input
        if (e.target.id !== 'fileInput' && e.target.id !== 'browseBtn') {
            fileInput.click();
        }
    });

    document.getElementById('browseBtn').addEventListener('click', () => {
        fileInput.click(); // Ensure browse button triggers file input
    });

    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#3498db';
            dropZone.style.backgroundColor = 'rgba(52, 152, 219, 0.1)';
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#aaa';
            dropZone.style.backgroundColor = 'transparent';
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            updateFileDisplay(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            updateFileDisplay(fileInput.files[0]);
        }
    });

    // Summarize Button Click Handler
    summarizeBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert("Please select a file to summarize.");
            return;
        }
        // Validate file extension
        if (!['.pdf', '.txt'].some(ext => file.name.toLowerCase().endsWith(ext))) {
            alert("Only .pdf or .txt files are supported.");
            return;
        }

        loadingOverlay.style.display = 'flex'; // Show loading spinner

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/summarize', {
                method: 'POST',
                body: formData
            });

            const result = await response.json(); // Parse JSON response
            if (response.ok) { // Check for successful HTTP status (2xx)
                // No need to store summaryText or downloadFilename in localStorage as we are redirecting directly
                // to the saved page, which will fetch the list of summaries from the backend.
                alert("Document summarized and saved successfully!"); // Provide feedback before redirect
                window.location.href = '/saved'; // Redirect to saved summaries page
            } else {
                // If response is not ok, throw an error with the message from backend
                throw new Error(result.error || 'Unknown error during summarization.');
            }

        } catch (err) {
            console.error("Summarization Error:", err);
            alert("Error: " + err.message); // Display error to user
        } finally {
            loadingOverlay.style.display = 'none'; // Hide loading spinner
        }
    });
}

// --- Summary Page Logic (REMOVED as per user clarification) ---
// The block starting with 'if (window.location.pathname === '/summary')' is removed.

// --- Saved Summaries Page Logic ---
// This runs when the saved.html page is loaded
if (window.location.pathname === '/saved') {
    document.addEventListener('DOMContentLoaded', async () => {
        const listContainer = document.getElementById('summaryList');
        try {
            const res = await fetch('/list_summaries');
            const files = await res.json(); // Get array of filenames (e.g., ["file1_summary.txt", "file2_summary.pdf"])

            if (files.length === 0) {
                listContainer.innerHTML = '<p class="empty-message">No saved summaries yet.</p>';
            } else {
                listContainer.innerHTML = ''; // Clear "Loading summaries..." or existing content
                files.forEach(filename => {
                    const item = document.createElement('div');
                    item.className = 'summary-item';

                    // Determine file type for icon and text
                    const isPdf = filename.toLowerCase().endsWith('.pdf');
                    const fileTypeIcon = isPdf ? '📄' : '📝'; // PDF icon vs Text icon
                    const viewText = isPdf ? 'Download PDF' : 'Download TXT';

                    item.innerHTML = `
                        <div class="summary-info">
                            <h3>${fileTypeIcon} ${filename}</h3>
                            <p class="summary-date">Saved: ${new Date().toLocaleString()}</p>
                            </div>
                        <div class="summary-actions">
                            <a href="/summaries/${filename}" class="action-btn view-btn" download="${filename}">${viewText}</a>
                        </div>
                    `;
                    listContainer.appendChild(item);
                });
            }
        } catch (error) {
            console.error('Error fetching saved summaries:', error);
            listContainer.innerHTML = `<p class="empty-message">Failed to load summaries.</p>`;
        }
    });
}

// --- Mobile Menu Toggle (remains unchanged) ---
const mobileMenuButton = document.querySelector('.mobile-menu');
const navLinks = document.querySelector('.nav-links');

if (mobileMenuButton && navLinks) {
    mobileMenuButton.addEventListener('click', () => {
        navLinks.classList.toggle('active');
    });
}