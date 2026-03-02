document.addEventListener("DOMContentLoaded", () => {
    // --- Theme ---
    const themeToggle = document.getElementById("theme-toggle");
    const root = document.documentElement;

    function applyTheme(theme) {
        root.setAttribute("data-theme", theme);
        localStorage.setItem("theme", theme);
    }

    // Load saved theme or detect system preference
    const saved = localStorage.getItem("theme");
    if (saved) {
        applyTheme(saved);
    } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        applyTheme("dark");
    }

    themeToggle.addEventListener("click", () => {
        const current = root.getAttribute("data-theme");
        applyTheme(current === "dark" ? "light" : "dark");
    });

    // --- State ---
    let currentFile = null; // { file_id, filename, stored_as }
    let separatedTracks = null; // { trackName: relativePath, ... }
    let mixFiles = []; // [{ path, label }, ...]

    // --- DOM elements ---
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadProgressBar = document.getElementById("upload-progress");
    const fileInfo = document.getElementById("file-info");
    const loadedFilename = document.getElementById("loaded-filename");
    const btnRemoveFile = document.getElementById("btn-remove-file");
    const separationSection = document.getElementById("separation-section");
    const btnSeparate = document.getElementById("btn-separate");
    const separationProgress = document.getElementById("separation-progress");
    const tracksContainer = document.getElementById("tracks-container");
    const tracksList = document.getElementById("tracks-list");
    const transpositionSection = document.getElementById("transposition-section");
    const transposeSource = document.getElementById("transpose-source");
    const transposeSemitones = document.getElementById("transpose-semitones");
    const semitoneDisplay = document.getElementById("semitone-display");
    const semitoneLabel = document.getElementById("semitone-label");
    const btnTranspose = document.getElementById("btn-transpose");
    const transposeProgress = document.getElementById("transpose-progress");
    const transposeResult = document.getElementById("transpose-result");
    const transposeResultContent = document.getElementById("transpose-result-content");
    const btnMix = document.getElementById("btn-mix");
    const mixProgress = document.getElementById("mix-progress");
    const mixResult = document.getElementById("mix-result");
    const mixResultContent = document.getElementById("mix-result-content");
    const youtubeUrl = document.getElementById("youtube-url");
    const btnYoutube = document.getElementById("btn-youtube");
    const btnClearYoutube = document.getElementById("btn-clear-youtube");
    const youtubeProgress = document.getElementById("youtube-progress");

    // --- Drag & Drop ---
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    });

    dropZone.addEventListener("click", (e) => {
        if (e.target.closest('label[for="file-input"]')) return;
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });

    // --- Upload ---
    function uploadFile(file) {
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            showError(uploadProgressBar, "Le fichier dépasse la limite de 50 Mo.");
            return;
        }

        const ext = file.name.split(".").pop().toLowerCase();
        const allowedExts = ["mp3", "wav", "flac", "ogg", "m4a", "aac", "opus", "webm", "3gp", "weba"];
        const isAudioMime = file.type && file.type.startsWith("audio/");
        if (!allowedExts.includes(ext) && !isAudioMime) {
            showError(uploadProgressBar, "Format non supporté. Utilisez MP3, WAV, FLAC, OGG, M4A ou AAC.");
            return;
        }

        const progressFill = uploadProgressBar.querySelector(".progress-fill");
        const progressText = uploadProgressBar.querySelector(".progress-text");

        uploadProgressBar.hidden = false;
        progressFill.classList.remove("indeterminate");
        progressFill.style.width = "0%";
        progressFill.style.background = "";
        progressText.textContent = `${file.name} — 0 %`;

        const formData = new FormData();
        formData.append("file", file);

        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener("progress", (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                progressFill.style.width = pct + "%";
                const sizeMB = (e.total / (1024 * 1024)).toFixed(1);
                progressText.textContent = `${file.name} — ${pct} % de ${sizeMB} Mo`;
            }
        });

        xhr.addEventListener("load", () => {
            try {
                const data = JSON.parse(xhr.responseText);
                if (xhr.status >= 200 && xhr.status < 300) {
                    progressFill.style.width = "100%";
                    progressText.textContent = `${file.name} — Chargé !`;
                    currentFile = data;
                    setTimeout(() => {
                        uploadProgressBar.hidden = true;
                        showFileLoaded();
                    }, 600);
                } else {
                    showError(uploadProgressBar, data.error || "Erreur lors de l'envoi.");
                }
            } catch {
                showError(uploadProgressBar, "Erreur lors de l'envoi.");
            }
        });

        xhr.addEventListener("error", () => {
            showError(uploadProgressBar, "Erreur de connexion au serveur.");
        });

        xhr.open("POST", "/upload");
        xhr.send(formData);
    }

    function showFileLoaded() {
        fileInfo.hidden = false;
        loadedFilename.textContent = `Fichier chargé : ${currentFile.filename}`;
        separationSection.hidden = false;
        transpositionSection.hidden = false;
        tracksContainer.hidden = true;
        separatedTracks = null;
        mixFiles = [];
        transposeResult.hidden = true;
        resetTransposeSource();
        updateSemitoneLabel();
    }

    btnRemoveFile.addEventListener("click", () => {
        currentFile = null;
        separatedTracks = null;
        mixFiles = [];
        fileInfo.hidden = true;
        separationSection.hidden = true;
        transpositionSection.hidden = true;
        tracksContainer.hidden = true;
        transposeResult.hidden = true;
        uploadProgressBar.hidden = true;
        fileInput.value = "";
    });

    // --- YouTube Import ---
    youtubeUrl.addEventListener("input", () => {
        btnClearYoutube.hidden = !youtubeUrl.value;
    });

    btnClearYoutube.addEventListener("click", () => {
        youtubeUrl.value = "";
        btnClearYoutube.hidden = true;
        youtubeUrl.focus();
    });

    btnYoutube.addEventListener("click", async () => {
        const url = youtubeUrl.value.trim();
        if (!url) return;

        const ytPattern = /^https?:\/\/(www\.|m\.)?(youtube\.com\/watch\?v=|youtu\.be\/|music\.youtube\.com\/watch\?v=)[\w-]+/;
        if (!ytPattern.test(url)) {
            showError(youtubeProgress, "URL YouTube invalide.");
            youtubeProgress.hidden = false;
            return;
        }

        btnYoutube.disabled = true;
        youtubeProgress.hidden = false;
        const progressFill = youtubeProgress.querySelector(".progress-fill");
        const progressText = youtubeProgress.querySelector(".progress-text");
        progressFill.classList.add("indeterminate");
        progressFill.style.background = "";
        progressText.textContent = "Téléchargement depuis YouTube...";

        try {
            const response = await fetch("/import_youtube", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url }),
            });
            const data = await response.json();

            if (!response.ok) {
                showError(youtubeProgress, data.error || "Erreur.");
                btnYoutube.disabled = false;
                return;
            }

            pollYoutubeJob(data.job_id, progressText, progressFill);
        } catch (err) {
            showError(youtubeProgress, "Erreur de connexion au serveur.");
            btnYoutube.disabled = false;
        }
    });

    async function pollYoutubeJob(jobId, progressText, progressFill) {
        try {
            const response = await fetch(`/job_status/${jobId}`);
            const data = await response.json();

            if (data.status === "running") {
                progressText.textContent = data.progress || "Téléchargement...";
                setTimeout(() => pollYoutubeJob(jobId, progressText, progressFill), 1500);
            } else if (data.status === "done") {
                progressFill.classList.remove("indeterminate");
                progressFill.style.width = "100%";
                progressText.textContent = "Importé !";
                currentFile = {
                    file_id: data.file_id,
                    filename: data.filename,
                    stored_as: data.stored_as,
                };
                showFileLoaded();
                btnYoutube.disabled = false;
                setTimeout(() => { youtubeProgress.hidden = true; }, 1500);
            } else if (data.status === "error") {
                showError(youtubeProgress, data.error || "Erreur lors du téléchargement.");
                progressFill.classList.remove("indeterminate");
                btnYoutube.disabled = false;
            }
        } catch (err) {
            showError(youtubeProgress, "Erreur de connexion.");
            btnYoutube.disabled = false;
        }
    }

    // --- Separation ---
    btnSeparate.addEventListener("click", async () => {
        if (!currentFile) return;

        btnSeparate.disabled = true;
        separationProgress.hidden = false;
        tracksContainer.hidden = true;
        const progressFill = separationProgress.querySelector(".progress-fill");
        const progressText = separationProgress.querySelector(".progress-text");
        progressFill.classList.add("indeterminate");
        progressFill.style.width = "";

        try {
            const response = await fetch("/separate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    stored_as: currentFile.stored_as,
                    model: document.getElementById("model-select").value,
                }),
            });
            const data = await response.json();

            if (!response.ok) {
                showError(separationProgress, data.error || "Erreur.");
                btnSeparate.disabled = false;
                return;
            }

            const jobId = data.job_id;
            pollJob(jobId, progressText, progressFill);
        } catch (err) {
            showError(separationProgress, "Erreur de connexion au serveur.");
            btnSeparate.disabled = false;
        }
    });

    async function pollJob(jobId, progressText, progressFill) {
        try {
            const response = await fetch(`/job_status/${jobId}`);
            const data = await response.json();

            if (data.status === "running") {
                progressText.textContent = data.progress || "En cours...";
                setTimeout(() => pollJob(jobId, progressText, progressFill), 2000);
            } else if (data.status === "done") {
                progressFill.classList.remove("indeterminate");
                progressFill.style.width = "100%";
                progressText.textContent = "Terminé !";
                separatedTracks = data.tracks;
                displayTracks(data.tracks);
                updateTransposeSourceOptions();
                btnSeparate.disabled = false;
                setTimeout(() => {
                    separationProgress.hidden = true;
                }, 1500);
            } else if (data.status === "error") {
                showError(separationProgress, data.error || "Erreur lors de la séparation.");
                progressFill.classList.remove("indeterminate");
                btnSeparate.disabled = false;
            }
        } catch (err) {
            showError(separationProgress, "Erreur de connexion.");
            btnSeparate.disabled = false;
        }
    }

    const TRACK_LABELS = {
        vocals: "Voix",
        drums: "Batterie",
        bass: "Basse",
        guitar: "Guitare",
        piano: "Piano",
        other: "Autres instruments",
        instrumental: "Instrumental (sans voix)",
    };

    function displayTracks(tracks) {
        tracksList.innerHTML = "";
        tracksContainer.hidden = false;
        mixResult.hidden = true;

        const order = ["vocals", "instrumental", "drums", "bass", "guitar", "piano", "other"];
        const sortedKeys = Object.keys(tracks).sort(
            (a, b) => order.indexOf(a) - order.indexOf(b)
        );

        for (const name of sortedKeys) {
            const path = tracks[name];
            const item = document.createElement("div");
            item.className = "track-item";

            const label = TRACK_LABELS[name] || name;

            item.innerHTML = `
                <input type="checkbox" class="track-checkbox" data-track-path="${path}" data-track-name="${name}" title="Sélectionner pour l'assemblage">
                <span class="track-name">${label}</span>
                <audio controls preload="none" src="/download/${path}"></audio>
                <div class="track-actions">
                    <a href="/download/${path}" class="btn btn-small btn-success" download>Télécharger</a>
                </div>
            `;
            tracksList.appendChild(item);
        }

        updateMixButton();
    }

    function updateMixButton() {
        const checked = tracksList.querySelectorAll(".track-checkbox:checked");
        btnMix.disabled = checked.length < 2;
        if (checked.length >= 2) {
            btnMix.textContent = `Assembler ${checked.length} pistes sélectionnées`;
        } else {
            btnMix.textContent = "Assembler les pistes sélectionnées";
        }
    }

    tracksList.addEventListener("change", (e) => {
        if (e.target.classList.contains("track-checkbox")) {
            updateMixButton();
        }
    });

    btnMix.addEventListener("click", async () => {
        const checked = tracksList.querySelectorAll(".track-checkbox:checked");
        if (checked.length < 2) return;

        const trackPaths = Array.from(checked).map(cb => cb.dataset.trackPath);

        btnMix.disabled = true;
        mixProgress.hidden = false;
        mixResult.hidden = true;

        try {
            const response = await fetch("/mix", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ tracks: trackPaths }),
            });
            const data = await response.json();

            mixProgress.hidden = true;

            if (!response.ok) {
                showError(mixResultContent, data.error || "Erreur lors de l'assemblage.");
                mixResult.hidden = false;
                btnMix.disabled = false;
                return;
            }

            mixResult.hidden = false;
            const names = Array.from(checked).map(cb => TRACK_LABELS[cb.dataset.trackName] || cb.dataset.trackName);
            const mixLabel = "Mix : " + names.join(" + ");
            mixResultContent.innerHTML = `
                <div class="track-item">
                    <span class="track-name">${mixLabel}</span>
                    <audio controls preload="none" src="/download/${data.mix_file}"></audio>
                    <div class="track-actions">
                        <a href="/download/${data.mix_file}" class="btn btn-small btn-success" download>Télécharger</a>
                    </div>
                </div>
            `;
            mixFiles.push({ path: data.mix_file, label: mixLabel });
            updateTransposeSourceOptions();
            transposeSource.value = data.mix_file;
            updateMixButton();
        } catch (err) {
            mixProgress.hidden = true;
            showError(mixResultContent, "Erreur de connexion au serveur.");
            mixResult.hidden = false;
            btnMix.disabled = false;
        }
    });

    // --- Transposition ---
    function resetTransposeSource() {
        transposeSource.innerHTML = '<option value="original">Fichier original</option>';
    }

    function updateTransposeSourceOptions() {
        resetTransposeSource();
        if (separatedTracks) {
            const order = ["vocals", "instrumental", "drums", "bass", "guitar", "piano", "other"];
            const sortedKeys = Object.keys(separatedTracks).sort(
                (a, b) => order.indexOf(a) - order.indexOf(b)
            );
            for (const name of sortedKeys) {
                const label = TRACK_LABELS[name] || name;
                const opt = document.createElement("option");
                opt.value = separatedTracks[name];
                opt.textContent = label;
                transposeSource.appendChild(opt);
            }
        }
        for (const mix of mixFiles) {
            const opt = document.createElement("option");
            opt.value = mix.path;
            opt.textContent = mix.label;
            transposeSource.appendChild(opt);
        }
    }

    transposeSemitones.addEventListener("input", () => {
        updateSemitoneLabel();
    });

    function updateSemitoneLabel() {
        const val = parseInt(transposeSemitones.value);
        const sign = val > 0 ? "+" : "";
        semitoneDisplay.textContent = `${sign}${val}`;

        if (val === 0) {
            semitoneLabel.textContent = "Pas de transposition";
            btnTranspose.disabled = true;
        } else {
            btnTranspose.disabled = false;
            const abs = Math.abs(val);
            const direction = val > 0 ? "vers le haut" : "vers le bas";

            if (abs === 12) {
                semitoneLabel.textContent = `1 octave (6 tons) ${direction}`;
            } else if (abs % 2 === 0) {
                const tons = abs / 2;
                semitoneLabel.textContent = `${tons} ton${tons > 1 ? "s" : ""} ${direction}`;
            } else if (abs === 1) {
                semitoneLabel.textContent = `1/2 ton ${direction}`;
            } else {
                const tons = Math.floor(abs / 2);
                if (tons === 0) {
                    semitoneLabel.textContent = `1/2 ton ${direction}`;
                } else {
                    semitoneLabel.textContent = `${tons} ton${tons > 1 ? "s" : ""} 1/2 ${direction}`;
                }
            }
        }
    }

    btnTranspose.addEventListener("click", async () => {
        if (!currentFile) return;

        const semitones = parseInt(transposeSemitones.value);
        if (semitones === 0) return;

        const sourceVal = transposeSource.value;
        let source;
        if (sourceVal === "original") {
            source = currentFile.stored_as;
        } else {
            source = sourceVal;
        }

        btnTranspose.disabled = true;
        transposeProgress.hidden = false;
        transposeResult.hidden = true;

        try {
            const response = await fetch("/transpose", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source, semitones }),
            });
            const data = await response.json();

            transposeProgress.hidden = true;

            if (!response.ok) {
                showError(transposeResultContent, data.error || "Erreur.");
                transposeResult.hidden = false;
                btnTranspose.disabled = false;
                return;
            }

            transposeResult.hidden = false;
            transposeResultContent.innerHTML = `
                <div class="track-item">
                    <span class="track-name">${data.filename}</span>
                    <audio controls preload="none" src="/download/${data.transposed_file}"></audio>
                    <div class="track-actions">
                        <a href="/download/${data.transposed_file}" class="btn btn-small btn-success" download>Télécharger</a>
                    </div>
                </div>
            `;
            btnTranspose.disabled = false;
        } catch (err) {
            transposeProgress.hidden = true;
            showError(transposeResultContent, "Erreur de connexion au serveur.");
            transposeResult.hidden = false;
            btnTranspose.disabled = false;
        }
    });

    // --- Helpers ---
    function showError(container, message) {
        if (container.classList.contains("progress-bar")) {
            const text = container.querySelector(".progress-text");
            const fill = container.querySelector(".progress-fill");
            if (text) text.textContent = message;
            if (fill) {
                fill.classList.remove("indeterminate");
                fill.style.width = "0%";
                fill.style.background = "var(--danger)";
            }
        } else {
            container.hidden = false;
            container.innerHTML = `<div class="error-message">${message}</div>`;
        }
    }

    // Initialize
    updateSemitoneLabel();
});
