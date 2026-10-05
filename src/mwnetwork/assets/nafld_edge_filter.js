// Client-side interactivity for the exported NAFLD subgraph plot
// (nafld_subgraph.export_nafld_html): a floating control panel with
//   - one checkbox per NAFLD dataset ({study_id}_{label_name}) -- a node is
//     shown if it was tested in ANY checked dataset (union, not intersection),
//     since a metabolite is routinely tested in more than one study.
//   - a radio to color nodes either by super_class (one trace per class, as
//     in the core graph viewer, legend click still toggles a class), by
//     nafld_kendall_tau (single trace, continuous RdYlGn_r colorscale), or by
//     which single NAFLD study tested them (one trace per study, legend
//     click toggles a study same as super_class mode; a node tested in more
//     than one study is grouped as "shared" and drawn white instead).
//
// Trace layout: [0]=positive-r edges, [1]=negative-r edges,
// [2 .. 2+NUM_SUPER_CLASSES)=one trace per super_class,
// [2+NUM_SUPER_CLASSES]=the single tau-colored trace,
// [3+NUM_SUPER_CLASSES .. 3+NUM_SUPER_CLASSES+NUM_STUDY_GROUPS)=one trace
// per study (+ one "shared" trace).
//
// Node data (position/size/tau/hover text) lives in flat, dataset-filter-
// independent arrays (nodeX/nodeY/...), indexed 0..N-1 in the same order the
// per-super_class traces were built in (classNodeIndices/studyNodeIndices
// record which global indices belong to which trace) -- so a dataset-filter
// change just rebuilds x/y/text subsets per trace via Plotly.restyle, no
// reliance on reading back Plotly's own current state.
//
// __PLACEHOLDER__ tokens substituted with per-run data by Python before
// injection; '{plot_id}' is plotly's own placeholder (write_html).

var gd = document.getElementById('{plot_id}');

var N = __N__;
var nodeSuperCode = __NODE_SUPERCODE__;
var nodeX = __NODE_X__;
var nodeY = __NODE_Y__;
var nodeSize = __NODE_SIZE__;
var nodeTau = __NODE_TAU__;
var nodeHover = __NODE_HOVER__;
var nodeDatasetIdx = __NODE_DATASET_IDX__; // array (len N) of arrays of dataset indices
var datasetIds = __DATASET_IDS__;
var classNodeIndices = __CLASS_NODE_INDICES__; // array (len NUM_SUPER_CLASSES) of arrays of global node indices
var NUM_SUPER_CLASSES = __NUM_SUPER_CLASSES__;
var nodeStudyCode = __NODE_STUDYCODE__;
var studyNodeIndices = __STUDY_NODE_INDICES__; // array (len NUM_STUDY_GROUPS) of arrays of global node indices
var NUM_STUDY_GROUPS = __NUM_STUDY_GROUPS__;

var edgeU = __EDGE_U__;
var edgeV = __EDGE_V__;
var edgeX0 = __EDGE_X0__;
var edgeY0 = __EDGE_Y0__;
var edgeX1 = __EDGE_X1__;
var edgeY1 = __EDGE_Y1__;
var edgeSign = __EDGE_SIGN__;

var NUM_EDGE_TRACES = 2;
var CLASS_TRACE_START = NUM_EDGE_TRACES;
var TAU_TRACE_INDEX = NUM_EDGE_TRACES + NUM_SUPER_CLASSES;
var STUDY_TRACE_START = TAU_TRACE_INDEX + 1;

var superVisible = new Array(NUM_SUPER_CLASSES).fill(true);
var studyVisible = new Array(NUM_STUDY_GROUPS).fill(true);
var selectedDatasets = new Array(datasetIds.length).fill(true);
var colorMode = 'super_class'; // 'super_class' | 'tau' | 'study'

function passesDatasetFilter(i) {
    var ds = nodeDatasetIdx[i];
    for (var k = 0; k < ds.length; k++) {
        if (selectedDatasets[ds[k]]) return true;
    }
    return false;
}

function nodeVisible(i) {
    if (!passesDatasetFilter(i)) return false;
    if (colorMode === 'super_class') return superVisible[nodeSuperCode[i]];
    if (colorMode === 'study') return studyVisible[nodeStudyCode[i]];
    return true;
}

function recomputeGroupTraces(groupNodeIndices, numGroups, traceStart) {
    var xs = [], ys = [], texts = [], traceIdx = [];
    for (var code = 0; code < numGroups; code++) {
        var idxs = groupNodeIndices[code];
        var x = [], y = [], t = [];
        for (var j = 0; j < idxs.length; j++) {
            var i = idxs[j];
            if (!passesDatasetFilter(i)) continue;
            x.push(nodeX[i]); y.push(nodeY[i]); t.push(nodeHover[i]);
        }
        xs.push(x); ys.push(y); texts.push(t);
        traceIdx.push(traceStart + code);
    }
    Plotly.restyle(gd, {x: xs, y: ys, text: texts}, traceIdx);
}

function recomputeTauTrace() {
    var x = [], y = [], t = [], c = [], s = [];
    for (var i = 0; i < N; i++) {
        if (!passesDatasetFilter(i)) continue;
        x.push(nodeX[i]); y.push(nodeY[i]); t.push(nodeHover[i]); c.push(nodeTau[i]); s.push(nodeSize[i]);
    }
    Plotly.restyle(gd, {x: [x], y: [y], text: [t], 'marker.color': [c], 'marker.size': [s]}, [TAU_TRACE_INDEX]);
}

function recomputeEdges() {
    var vis = new Array(N);
    for (var i = 0; i < N; i++) vis[i] = nodeVisible(i);
    var xPos = [], yPos = [], xNeg = [], yNeg = [];
    for (var k = 0; k < edgeU.length; k++) {
        if (vis[edgeU[k]] && vis[edgeV[k]]) {
            var X = edgeSign[k] ? xPos : xNeg;
            var Y = edgeSign[k] ? yPos : yNeg;
            X.push(edgeX0[k], edgeX1[k], null);
            Y.push(edgeY0[k], edgeY1[k], null);
        }
    }
    Plotly.restyle(gd, {x: [xPos, xNeg], y: [yPos, yNeg]}, [0, 1]);
}

function applyTraceVisibility() {
    var classIdx = [], studyIdx = [];
    for (var i = 0; i < NUM_SUPER_CLASSES; i++) classIdx.push(CLASS_TRACE_START + i);
    for (var i = 0; i < NUM_STUDY_GROUPS; i++) studyIdx.push(STUDY_TRACE_START + i);

    if (colorMode === 'super_class') {
        Plotly.restyle(gd, {visible: superVisible.map(function (v) { return v ? true : 'legendonly'; })}, classIdx);
        Plotly.restyle(gd, {visible: [false]}, [TAU_TRACE_INDEX]);
        Plotly.restyle(gd, {visible: false}, studyIdx);
    } else if (colorMode === 'tau') {
        Plotly.restyle(gd, {visible: false}, classIdx);
        Plotly.restyle(gd, {visible: [true]}, [TAU_TRACE_INDEX]);
        Plotly.restyle(gd, {visible: false}, studyIdx);
    } else {
        Plotly.restyle(gd, {visible: false}, classIdx);
        Plotly.restyle(gd, {visible: [false]}, [TAU_TRACE_INDEX]);
        Plotly.restyle(gd, {visible: studyVisible.map(function (v) { return v ? true : 'legendonly'; })}, studyIdx);
    }
}

function redrawAll() {
    applyTraceVisibility();
    recomputeGroupTraces(classNodeIndices, NUM_SUPER_CLASSES, CLASS_TRACE_START);
    recomputeGroupTraces(studyNodeIndices, NUM_STUDY_GROUPS, STUDY_TRACE_START);
    recomputeTauTrace();
    recomputeEdges();
}

gd.on('plotly_legendclick', function (evt) {
    if (colorMode === 'super_class') {
        var code = evt.curveNumber - CLASS_TRACE_START;
        if (code < 0 || code >= NUM_SUPER_CLASSES) return true;
        superVisible[code] = !superVisible[code];
    } else if (colorMode === 'study') {
        var code2 = evt.curveNumber - STUDY_TRACE_START;
        if (code2 < 0 || code2 >= NUM_STUDY_GROUPS) return true;
        studyVisible[code2] = !studyVisible[code2];
    } else {
        return true;
    }
    applyTraceVisibility();
    recomputeEdges();
    return false;
});

gd.on('plotly_legenddoubleclick', function (evt) {
    if (colorMode === 'super_class') {
        var code = evt.curveNumber - CLASS_TRACE_START;
        if (code < 0 || code >= NUM_SUPER_CLASSES) return true;
        var onlyThisVisible = superVisible.every(function (v, i) { return i === code ? v : !v; });
        for (var i = 0; i < NUM_SUPER_CLASSES; i++) {
            superVisible[i] = onlyThisVisible ? true : (i === code);
        }
    } else if (colorMode === 'study') {
        var code2 = evt.curveNumber - STUDY_TRACE_START;
        if (code2 < 0 || code2 >= NUM_STUDY_GROUPS) return true;
        var onlyThisVisible2 = studyVisible.every(function (v, i) { return i === code2 ? v : !v; });
        for (var i = 0; i < NUM_STUDY_GROUPS; i++) {
            studyVisible[i] = onlyThisVisible2 ? true : (i === code2);
        }
    } else {
        return true;
    }
    applyTraceVisibility();
    recomputeEdges();
    return false;
});

// --- floating control panel: color-mode radio + per-dataset checkboxes ---
// Fixed to the viewport (not absolute inside gd's parent), so it sits
// outside the plot itself, pinned to the bottom-right corner of the page
// rather than overlapping the figure's legend/nodes.
//
// The name `controlPanel` is the contract assets/viewer_controls.js looks
// for: running after this script, it prepends its own edge toggle and search
// box into this panel rather than creating a second floating box that would
// land on top of this one.
var controlPanel = document.createElement('div');
controlPanel.style.cssText = 'position:fixed; bottom:10px; right:10px; z-index:1000; background:white; ' +
    'border:1px solid #ccc; border-radius:6px; padding:10px 14px; font-family:sans-serif; ' +
    'font-size:12px; max-height:60vh; overflow-y:auto; box-shadow:0 1px 4px rgba(0,0,0,0.2);';

var colorTitle = document.createElement('div');
colorTitle.textContent = 'Color nodes by';
colorTitle.style.cssText = 'font-weight:bold; margin-bottom:4px;';
controlPanel.appendChild(colorTitle);

[['super_class', 'super_class'], ['tau', 'Kendall tau (weighted)'], ['study', 'study (shared = white)']].forEach(function (pair, i) {
    var mode = pair[0], label_text = pair[1];
    var label = document.createElement('label');
    label.style.display = 'block';
    var radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'nafld_color_mode';
    radio.value = mode;
    radio.checked = i === 0;
    radio.onchange = function () {
        colorMode = mode;
        redrawAll();
    };
    label.appendChild(radio);
    label.appendChild(document.createTextNode(' ' + label_text));
    controlPanel.appendChild(label);
});

var dsTitle = document.createElement('div');
dsTitle.textContent = 'Datasets (shown if tested in ANY checked)';
dsTitle.style.cssText = 'font-weight:bold; margin:10px 0 4px;';
controlPanel.appendChild(dsTitle);

var toggleAll = document.createElement('div');
toggleAll.style.marginBottom = '4px';
var selectAllBtn = document.createElement('a');
selectAllBtn.textContent = 'all';
selectAllBtn.href = '#';
selectAllBtn.style.marginRight = '10px';
var selectNoneBtn = document.createElement('a');
selectNoneBtn.textContent = 'none';
selectNoneBtn.href = '#';
toggleAll.appendChild(selectAllBtn);
toggleAll.appendChild(selectNoneBtn);
controlPanel.appendChild(toggleAll);

var checkboxes = [];
datasetIds.forEach(function (dsId, i) {
    var label = document.createElement('label');
    label.style.display = 'block';
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.onchange = function () {
        selectedDatasets[i] = cb.checked;
        redrawAll();
    };
    checkboxes.push(cb);
    label.appendChild(cb);
    label.appendChild(document.createTextNode(' ' + dsId));
    controlPanel.appendChild(label);
});

selectAllBtn.onclick = function (e) {
    e.preventDefault();
    checkboxes.forEach(function (cb, i) { cb.checked = true; selectedDatasets[i] = true; });
    redrawAll();
};
selectNoneBtn.onclick = function (e) {
    e.preventDefault();
    checkboxes.forEach(function (cb, i) { cb.checked = false; selectedDatasets[i] = false; });
    redrawAll();
};

document.body.appendChild(controlPanel);
