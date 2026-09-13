const API_BASE = "http://127.0.0.1:8000";

let selectedFile = null;
let currentDetections = [];
let currentWorkers = [];
let currentImage = null;

const imageInput = document.getElementById("imageInput");
const browseBtn = document.getElementById("browseBtn");
const uploadZone = document.getElementById("uploadZone");
const uploadContent = document.getElementById("uploadContent");
const canvasWrapper = document.getElementById("canvasWrapper");
const canvas = document.getElementById("detectionCanvas");
const ctx = canvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");
const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("questionInput");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");


/* ---------------------------------------------------------
   API HEALTH
--------------------------------------------------------- */

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);

        if (!response.ok) {
            throw new Error("API unavailable");
        }

        statusDot.className = "status-dot online";
        statusText.textContent = "API ONLINE";

    } catch (error) {

        statusDot.className = "status-dot offline";
        statusText.textContent = "API OFFLINE";
    }
}

checkHealth();
setInterval(checkHealth, 10000);


/* ---------------------------------------------------------
   FILE UPLOAD
--------------------------------------------------------- */

browseBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    imageInput.click();
});

uploadZone.addEventListener("click", () => {
    if (!selectedFile) {
        imageInput.click();
    }
});

imageInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (file) {
        handleFile(file);
    }
});


uploadZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadZone.classList.add("dragging");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragging");
});

uploadZone.addEventListener("drop", (event) => {

    event.preventDefault();

    uploadZone.classList.remove("dragging");

    const file = event.dataTransfer.files[0];

    if (file && file.type.startsWith("image/")) {
        handleFile(file);
    }
});


async function handleFile(file) {

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = () => {

        currentImage = new Image();

        currentImage.onload = () => {

            uploadContent.classList.add("hidden");
            canvasWrapper.classList.remove("hidden");

            document.getElementById("imageMeta").classList.remove("hidden");

            document.getElementById("imageName").textContent =
                file.name;

            document.getElementById("imageDimensions").textContent =
                `${currentImage.naturalWidth} × ${currentImage.naturalHeight}px`;

            drawImage([]);

        };

        currentImage.src = reader.result;
    };

    reader.readAsDataURL(file);

    await runDetection(file);
}


/* ---------------------------------------------------------
   DETECTION
--------------------------------------------------------- */

async function runDetection(file) {

    setLoading(true);

    const formData = new FormData();
    formData.append("image", file);

    try {

        const response = await fetch(
            `${API_BASE}/detect`,
            {
                method: "POST",
                body: formData
            }
        );

        if (!response.ok) {
            throw new Error(
                `Detection failed (${response.status})`
            );
        }

        const data = await response.json();

        currentDetections = data.detections || [];

        updateDetectionSummary(currentDetections);
        renderDetectionTable(currentDetections);

        if (currentImage) {
            drawImage(currentDetections);
        }

        /*
         * Detection endpoint gives us raw objects.
         * Run a default PPE question as well so that the
         * dashboard immediately gets worker-level evidence.
         */
        await runReasoning(
            file,
            "Are the workers wearing helmets?"
        );

    } catch (error) {

        console.error(error);

        alert(
            "Detection failed. Make sure FastAPI is running on port 8000."
        );

    } finally {

        setLoading(false);
    }
}


/* ---------------------------------------------------------
   DRAW DETECTIONS
--------------------------------------------------------- */

function drawImage(detections) {

    if (!currentImage) {
        return;
    }

    const maxWidth = Math.min(
        uploadZone.clientWidth - 20,
        1100
    );

    const maxHeight = 600;

    const scale = Math.min(
        maxWidth / currentImage.naturalWidth,
        maxHeight / currentImage.naturalHeight,
        1
    );

    const width = Math.round(
        currentImage.naturalWidth * scale
    );

    const height = Math.round(
        currentImage.naturalHeight * scale
    );

    canvas.width = width;
    canvas.height = height;

    ctx.drawImage(
        currentImage,
        0,
        0,
        width,
        height
    );

    const xScale = width / currentImage.naturalWidth;
    const yScale = height / currentImage.naturalHeight;

    detections.forEach((det) => {

        if (!det.bbox) {
            return;
        }

        const x1 = det.bbox.x1 * xScale;
        const y1 = det.bbox.y1 * yScale;
        const x2 = det.bbox.x2 * xScale;
        const y2 = det.bbox.y2 * yScale;

        const className = det.class_name || "object";
        const confidence = Number(det.confidence || 0);

        const isPerson =
            className.toLowerCase() === "person";

        ctx.lineWidth = isPerson ? 2.5 : 1.6;

        /*
         * Keep the visualization monochrome/industrial.
         * Person uses accent, PPE uses white/gray.
         */
        ctx.strokeStyle =
            isPerson ? "#b8ff4d" : "#dce5ed";

        ctx.strokeRect(
            x1,
            y1,
            x2 - x1,
            y2 - y1
        );

        const label =
            `${className} ${(confidence * 100).toFixed(0)}%`;

        ctx.font = "700 10px Arial";

        const textWidth =
            ctx.measureText(label).width;

        const labelHeight = 18;

        ctx.fillStyle =
            isPerson
                ? "rgba(184,255,77,0.92)"
                : "rgba(10,14,18,0.92)";

        ctx.fillRect(
            x1,
            Math.max(0, y1 - labelHeight),
            textWidth + 10,
            labelHeight
        );

        ctx.fillStyle =
            isPerson ? "#071006" : "#edf2f7";

        ctx.fillText(
            label,
            x1 + 5,
            Math.max(12, y1 - 6)
        );
    });
}


/* ---------------------------------------------------------
   DETECTION SUMMARY
--------------------------------------------------------- */

function updateDetectionSummary(detections) {

    const people = detections.filter(
        d => (d.class_name || "").toLowerCase() === "person"
    );

    document.getElementById("workerCount").textContent =
        people.length;

    document.getElementById("detectionCount").textContent =
        detections.length;

    document.getElementById("tableCount").textContent =
        `${detections.length} objects`;
}


/* ---------------------------------------------------------
   DETECTION TABLE
--------------------------------------------------------- */

function renderDetectionTable(detections) {

    const tbody =
        document.getElementById("detectionTable");

    if (!detections.length) {

        tbody.innerHTML = `
            <tr>
                <td colspan="4" class="empty-table">
                    No detections returned.
                </td>
            </tr>
        `;

        return;
    }

    tbody.innerHTML = detections.map((det) => {

        const confidence =
            Number(det.confidence || 0) * 100;

        const bbox = det.bbox || {};

        return `
            <tr>
                <td>${det.detection_id ?? "—"}</td>

                <td>
                    <b>${escapeHtml(det.class_name || "unknown")}</b>
                </td>

                <td class="confidence">
                    ${confidence.toFixed(1)}%
                </td>

                <td class="bbox">
                    [${round(bbox.x1)},
                     ${round(bbox.y1)},
                     ${round(bbox.x2)},
                     ${round(bbox.y2)}]
                </td>
            </tr>
        `;

    }).join("");
}


/* ---------------------------------------------------------
   ASK IMAGE
--------------------------------------------------------- */

askBtn.addEventListener("click", async () => {

    if (!selectedFile) {
        alert("Upload an inspection image first.");
        return;
    }

    const question =
        questionInput.value.trim();

    if (!question) {
        alert("Enter a question about the image.");
        return;
    }

    await runReasoning(
        selectedFile,
        question
    );
});


questionInput.addEventListener("keydown", (event) => {

    if (event.key === "Enter") {
        askBtn.click();
    }

});


document.querySelectorAll(".question-chip")
    .forEach((button) => {

        button.addEventListener("click", () => {

            questionInput.value =
                button.dataset.question;

            askBtn.click();
        });

    });


async function runReasoning(file, question) {

    askBtn.disabled = true;
    askBtn.textContent = "ANALYZING...";

    const formData = new FormData();

    formData.append("image", file);
    formData.append("question", question);

    try {

        const response = await fetch(
            `${API_BASE}/reason`,
            {
                method: "POST",
                body: formData
            }
        );

        if (!response.ok) {
            throw new Error(
                `Reasoning failed (${response.status})`
            );
        }

        const data = await response.json();

        renderReasoning(data);

    } catch (error) {

        console.error(error);

        alert(
            "Reasoning failed. Check the FastAPI terminal for details."
        );

    } finally {

        askBtn.disabled = false;
        askBtn.textContent = "ANALYZE →";
    }
}


/* ---------------------------------------------------------
   REASONING RENDER
--------------------------------------------------------- */

function renderReasoning(data) {

    console.log("FULL REASONING RESPONSE:", data);

    const reasoningEnvelope =
        data.reasoning || {};

    /*
     * FastAPI response structure:
     *
     * data.reasoning.intent
     * data.reasoning.reasoning.status
     * data.reasoning.reasoning.answer
     * data.reasoning.reasoning.confidence
     */
    const reasoning =
        reasoningEnvelope.reasoning || {};

    const status =
        reasoning.status ||
        "INSUFFICIENT_INFORMATION";

    const answer =
        reasoning.answer ||
        "The system could not generate a reasoning answer.";

    const confidence =
        Number(reasoning.confidence ?? 0);

    const workers =
        data.workers || [];

    currentWorkers = workers;

    const result =
        document.getElementById("reasoningResult");

    result.classList.remove("hidden");

    document.getElementById("resultStatus").textContent =
        prettyStatus(status);

    document.getElementById("answerText").textContent =
        answer;

    /*
     * Display the actual reasoning confidence.
     * For PARTIAL / INSUFFICIENT_INFORMATION, use
     * evidence-based wording rather than implying
     * calibrated probability.
     */
    if (
        status === "INSUFFICIENT_INFORMATION" ||
        status === "PARTIAL"
    ) {

        document.getElementById("resultConfidence")
            .textContent = "EVIDENCE-BASED";

    } else if (confidence > 0) {

        document.getElementById("resultConfidence")
            .textContent =
                `${(confidence * 100).toFixed(0)}%`;

    } else {

        document.getElementById("resultConfidence")
            .textContent = "EVIDENCE-BASED";
    }

    renderWorkerResults(workers);
    updatePpeOverview(workers);
    renderEvidenceGraph(workers);
}

/* ---------------------------------------------------------
   WORKER RESULTS
--------------------------------------------------------- */

function renderWorkerResults(workers) {

    const container =
        document.getElementById("workerResults");

    if (!workers.length) {

        container.innerHTML = `
            <div class="graph-empty">
                No worker-level evidence available.
            </div>
        `;

        return;
    }

    container.innerHTML = workers.map((worker, index) => {

        const workerId =
            worker.worker_id ??
            worker.id ??
            index + 1;

        const ppe =
            worker.ppe || {};

        const helmet =
            getPpeEntry(worker, "helmet", ppe);

        const vest =
            getPpeEntry(
                worker,
                "safety_vest",
                ppe
            );

        const gloves =
            getPpeEntry(
                worker,
                "gloves",
                ppe
            );

        return `
            <div class="worker-card">

                <div class="worker-card-top">
                    <span class="worker-id">
                        WORKER ${workerId}
                    </span>

                    <span class="worker-state">
                        EVIDENCE
                    </span>
                </div>

                <div class="worker-details">

                    <div>
                        Helmet:
                        <b class="${stateClass(helmet.state)}">
                            ${prettyState(helmet.state)}
                        </b>
                    </div>

                    <div>
                        Vest:
                        <b class="${stateClass(vest.state)}">
                            ${prettyState(vest.state)}
                        </b>
                    </div>

                    <div>
                        Gloves:
                        <b class="${stateClass(gloves.state)}">
                            ${prettyState(gloves.state)}
                        </b>
                    </div>

                </div>

            </div>
        `;

    }).join("");
}


/* ---------------------------------------------------------
   PPE OVERVIEW
--------------------------------------------------------- */

function updatePpeOverview(workers) {

    const types = [
        {
            key: "helmet",
            label: "Helmet",
            stateId: "helmetState",
            summaryId: "helmetSummary"
        },
        {
            key: "safety_vest",
            label: "Safety Vest",
            stateId: "vestState",
            summaryId: "vestSummary"
        },
        {
            key: "gloves",
            label: "Gloves",
            stateId: "glovesState",
            summaryId: "glovesSummary"
        }
    ];

    types.forEach((type) => {

        const states = workers.map((worker) => {

            const ppe =
                worker.ppe || {};

            return getPpeEntry(
                worker,
                type.key,
                ppe
            ).state;
        });

        const counts = {
            PRESENT: states.filter(s => s === "PRESENT").length,
            ABSENT: states.filter(s => s === "ABSENT").length,
            UNKNOWN: states.filter(s => s === "UNKNOWN").length
        };

        const stateElement =
            document.getElementById(type.stateId);

        const summaryElement =
            document.getElementById(type.summaryId);

        if (!states.length) {

            stateElement.textContent = "—";
            stateElement.className =
                "state-badge neutral";

            summaryElement.textContent =
                "Awaiting evidence";

            return;
        }

        let overallState = "PARTIAL";

        if (counts.PRESENT === states.length) {
            overallState = "PRESENT";
        } else if (counts.ABSENT === states.length) {
            overallState = "ABSENT";
        } else if (
            counts.UNKNOWN === states.length
        ) {
            overallState = "UNKNOWN";
        }

        stateElement.textContent =
            overallState;

        stateElement.className =
            `state-badge ${stateClass(overallState)}`;

        summaryElement.textContent =
            `${counts.PRESENT} present · ` +
            `${counts.ABSENT} absent · ` +
            `${counts.UNKNOWN} unknown`;
    });
}


/* ---------------------------------------------------------
   EVIDENCE GRAPH
--------------------------------------------------------- */

function renderEvidenceGraph(workers) {

    const container =
        document.getElementById("evidenceGraph");

    if (!workers.length) {

        container.innerHTML = `
            <div class="graph-empty">
                No worker-level evidence available.
            </div>
        `;

        return;
    }

    container.innerHTML = workers.map(
        (worker, index) => {

            const workerId =
                worker.worker_id ??
                worker.id ??
                index + 1;

            const ppe =
                worker.ppe || {};

            const helmet =
                getPpeEntry(
                    worker,
                    "helmet",
                    ppe
                );

            const vest =
                getPpeEntry(
                    worker,
                    "safety_vest",
                    ppe
                );

            const gloves =
                getPpeEntry(
                    worker,
                    "gloves",
                    ppe
                );

            return `
                <div class="graph-card">

                    <div class="graph-worker">
                        WORKER ${workerId}
                    </div>

                    ${buildChain(
                        "Worker",
                        "Head",
                        "Helmet",
                        helmet
                    )}

                    ${buildChain(
                        "Worker",
                        "Body",
                        "Safety Vest",
                        vest
                    )}

                    ${buildChain(
                        "Worker",
                        "Hands",
                        "Gloves",
                        gloves
                    )}

                </div>
            `;

        }
    ).join("");
}


function buildChain(
    first,
    second,
    third,
    entry
) {

    const state =
        entry.state || "UNKNOWN";

    return `
        <div style="margin-bottom:12px">

            <div class="graph-chain">

                <span class="graph-node">
                    ${first}
                </span>

                <span class="graph-arrow">→</span>

                <span class="graph-node">
                    ${second}
                </span>

                <span class="graph-arrow">→</span>

                <span class="graph-node">
                    ${third}
                </span>

            </div>

            <div class="graph-state">
                ${prettyState(state)}
                ${entry.association_mode
                    ? ` · ${entry.association_mode}`
                    : ""}
            </div>

        </div>
    `;
}


/* ---------------------------------------------------------
   CLEAR
--------------------------------------------------------- */

clearBtn.addEventListener("click", () => {

    selectedFile = null;
    currentDetections = [];
    currentWorkers = [];
    currentImage = null;

    imageInput.value = "";

    uploadContent.classList.remove("hidden");
    canvasWrapper.classList.add("hidden");

    document
        .getElementById("imageMeta")
        .classList.add("hidden");

    document
        .getElementById("reasoningResult")
        .classList.add("hidden");

    document.getElementById("workerCount")
        .textContent = "—";

    document.getElementById("detectionCount")
        .textContent = "—";

    document.getElementById("tableCount")
        .textContent = "0 objects";

    document.getElementById("detectionTable")
        .innerHTML = `
            <tr>
                <td colspan="4" class="empty-table">
                    No detections yet.
                </td>
            </tr>
        `;

    document.getElementById("evidenceGraph")
        .innerHTML = `
            <div class="graph-empty">
                Upload an image and run an analysis to inspect the evidence graph.
            </div>
        `;

    [
        ["helmetState", "helmetSummary"],
        ["vestState", "vestSummary"],
        ["glovesState", "glovesSummary"]
    ].forEach(([stateId, summaryId]) => {

        const state =
            document.getElementById(stateId);

        state.textContent = "—";
        state.className = "state-badge neutral";

        document.getElementById(summaryId)
            .textContent = "Awaiting image";
    });

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );
});


/* ---------------------------------------------------------
   HELPERS
--------------------------------------------------------- */

function getPpeEntry(worker, key, ppe) {

    if (ppe && ppe[key]) {
        return ppe[key];
    }

    if (worker && worker[key]) {
        return worker[key];
    }

    return {
        state: "UNKNOWN",
        confidence: 0,
        association_mode: "no_evidence"
    };
}


function prettyStatus(status) {

    return String(status)
        .replaceAll("_", " ")
        .replace("DETERMINED", "DETERMINED")
        .toUpperCase();
}


function prettyState(state) {

    return String(state || "UNKNOWN")
        .replaceAll("_", " ")
        .toUpperCase();
}


function stateClass(state) {

    switch (String(state).toUpperCase()) {

        case "PRESENT":
            return "present";

        case "ABSENT":
            return "absent";

        case "UNKNOWN":
            return "unknown";

        case "PARTIAL":
            return "partial";

        default:
            return "neutral";
    }
}


function round(value) {

    if (value === undefined || value === null) {
        return "—";
    }

    return Math.round(Number(value));
}


function escapeHtml(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function setLoading(loading) {

    if (loading) {

        statusText.textContent =
            "RUNNING INFERENCE";

    } else {

        checkHealth();
    }
}


/* ---------------------------------------------------------
   WINDOW RESIZE
--------------------------------------------------------- */

window.addEventListener("resize", () => {

    if (currentImage) {
        drawImage(currentDetections);
    }

});




