// Client-side edge filtering for the exported metabolite correlation
// network plot (see _workbench.ipynb, cell building `fig`).
//
// Plotly's legend click/double-click only toggles trace visibility -- it
// has no notion that an edge trace depends on which node traces (one per
// super_class) are currently visible. This script intercepts those legend
// events and rebuilds the two edge traces (positive/negative r), keeping
// only edges whose BOTH endpoint nodes belong to a currently active
// super_class.
//
// Visibility is tracked in our own `superVisible` array rather than read
// back from `gd.data[i].visible` after the click: Plotly applies its
// default legend toggle asynchronously (a redraw), and on a graph this
// size that redraw can still be in flight when we go to read it, so a
// naive "wait a bit, then read gd.data" approach reads stale state --
// symptom: edges update one click late. Tracking + applying visibility
// ourselves (and returning false to cancel Plotly's own default toggle)
// removes that race entirely.
//
// Injected into the exported html via fig.write_html(post_script=...):
// the `__PLACEHOLDER__` tokens below are substituted with per-run data
// (json.dumps'd arrays) by Python before injection; `{plot_id}` is plotly's
// own placeholder, substituted by write_html itself -- left untouched here.

var gd = document.getElementById('{plot_id}');
var N = __N__;
var nodeSuperCode = __NODE_SUPERCODE__;
// per-node position/size, in global node index order. Only the node-search
// script (assets/viewer_controls.js, appended after this one) reads these --
// edge filtering itself works off the pre-split edge endpoint arrays below.
var nodeX = __NODE_X__;
var nodeY = __NODE_Y__;
var nodeSize = __NODE_SIZE__;
var edgeU = __EDGE_U__;
var edgeV = __EDGE_V__;
var edgeX0 = __EDGE_X0__;
var edgeY0 = __EDGE_Y0__;
var edgeX1 = __EDGE_X1__;
var edgeY1 = __EDGE_Y1__;
var edgeSign = __EDGE_SIGN__;
var NUM_EDGE_TRACES = 2; // trace 0 = positive-r edges, trace 1 = negative-r edges
var NUM_SUPER_CLASSES = __NUM_SUPER_CLASSES__;

var superVisible = new Array(NUM_SUPER_CLASSES).fill(true);

// a node is shown iff its super_class is active -- the only filter this
// viewer has. Part of the contract assets/viewer_controls.js relies on.
function nodeVisible(i) {
    return superVisible[nodeSuperCode[i]];
}

function recomputeEdges() {
    var xPos = [], yPos = [], xNeg = [], yNeg = [];
    for (var k = 0; k < edgeU.length; k++) {
        var cu = nodeSuperCode[edgeU[k]], cv = nodeSuperCode[edgeV[k]];
        if (superVisible[cu] && superVisible[cv]) {
            var X = edgeSign[k] ? xPos : xNeg;
            var Y = edgeSign[k] ? yPos : yNeg;
            X.push(edgeX0[k], edgeX1[k], null);
            Y.push(edgeY0[k], edgeY1[k], null);
        }
    }
    Plotly.restyle(gd, {x: [xPos, xNeg], y: [yPos, yNeg]}, [0, 1]);
}

function applyNodeVisibility() {
    var visible = superVisible.map(function (v) { return v ? true : 'legendonly'; });
    var traceIdx = superVisible.map(function (_, i) { return NUM_EDGE_TRACES + i; });
    Plotly.restyle(gd, {visible: visible}, traceIdx);
}

gd.on('plotly_legendclick', function (evt) {
    var code = evt.curveNumber - NUM_EDGE_TRACES;
    if (code < 0 || code >= NUM_SUPER_CLASSES) return true;
    superVisible[code] = !superVisible[code];
    applyNodeVisibility();
    recomputeEdges();
    return false; // we already applied the toggle ourselves -- skip Plotly's default
});

gd.on('plotly_legenddoubleclick', function (evt) {
    var code = evt.curveNumber - NUM_EDGE_TRACES;
    if (code < 0 || code >= NUM_SUPER_CLASSES) return true;
    // mimic plotly's own isolate behavior: if this is already the sole
    // visible group, un-isolate (show all); otherwise isolate it alone
    var onlyThisVisible = superVisible.every(function (v, i) { return i === code ? v : !v; });
    for (var i = 0; i < NUM_SUPER_CLASSES; i++) {
        superVisible[i] = onlyThisVisible ? true : (i === code);
    }
    applyNodeVisibility();
    recomputeEdges();
    return false;
});
