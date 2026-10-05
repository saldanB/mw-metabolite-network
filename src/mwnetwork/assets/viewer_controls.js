// Shared view controls for the exported network viewers
// (network.export_html and nafld_subgraph.export_nafld_html): an edge
// show/hide toggle, and a search box that rings matching nodes.
//
// Edges start HIDDEN. At this graph's density (hundreds of thousands of
// edges over a few thousand nodes) drawing them all produces a solid mat
// that hides the very structure the layout is meant to show, and they are
// only readable once a super_class or dataset filter has cut the graph down.
// Because of that default, the Python side ships the two edge traces EMPTY
// and this script fills them from the edgeX0/edgeY0/... arrays on demand --
// so the exported html carries one copy of the edge coordinates instead of
// two, and first paint doesn't have to lay out every edge.
//
// Appended to the viewer's own edge-filter script, into the SAME script
// scope, so it can reuse what that script already defines rather than
// re-shipping it. The contract each host script must provide:
//     N              number of nodes
//     nodeX, nodeY   per-node position, in global node index order
//     nodeSize       per-node marker size, same order
//     nodeVisible(i) whether node i is currently shown
//     recomputeEdges() called by the host after every visibility change
//     controlPanel   OPTIONAL: an existing floating panel to prepend these
//                    controls into. The NAFLD viewer has one (its color-mode
//                    radio and dataset checkboxes); the core viewer does not,
//                    so one is created here with matching styling. Either way
//                    there is exactly ONE panel, in the same corner.
//
// Matching is case-insensitive substring over a per-node, per-field haystack
// built in Python (viewer_search.search_fields): RefMet id and name, formula,
// chemical class, and every cross-reference ID both bare and prefixed. So
// "C00031", "CHEBI:17234", "17234", "HMDB0000122", "C6H12O6" and "glucose"
// all find their node.
//
// Whitespace/comma/semicolon-separated terms are OR'd, so a pasted list of
// IDs lights all of them up at once. A term that itself contains a space --
// most metabolite names and every multi-word chemical class -- has to be
// quoted, otherwise it is read as two OR'd terms:
//     organic acids       "organic" OR "acids", so "fatty acids" matches too
//     "organic acids"     the phrase, as one term
// Either quote character works, and an unterminated one runs to the end of
// the query so the search stays sane mid-typing.
//
// A "?tag" token scopes the terms that follow it to one field:
//     c16h28              c16h28 anywhere
//     ?formula c16h28     c16h28 in the molecular formula only
//     ?kegg c00031        c00031 in the KEGG id only
//     ?name glucose ?formula c6h12o6
//                         either, i.e. the union of the two scoped searches
// Tags are resolved through SEARCH_FIELD_INDEX, which Python builds from its
// own field list (canonical tags plus aliases), so the vocabulary is defined
// in exactly one place. An unrecognized tag is reported rather than quietly
// demoted to an all-field search, which would silently return the wrong set.
//
// __PLACEHOLDER__ tokens are substituted with per-run data by Python before
// injection; '{plot_id}' is plotly's own placeholder (write_html).

var nodeSearchFields = __NODE_SEARCH__;        // per node: one haystack per field
var SEARCH_FIELD_TAGS = __SEARCH_FIELD_TAGS__;  // canonical tags, for the UI
var SEARCH_FIELD_INDEX = __SEARCH_FIELD_INDEX__;  // tag or alias -> field position
var HIGHLIGHT_TRACE_INDEX = __HIGHLIGHT_TRACE_INDEX__;

// starts false, matching the empty edge traces the Python side emits -- so
// no restyle is needed at load, only when the box is ticked
var edgesVisible = false;

// the ring is drawn concentric with the node marker, so it has to be bigger
// than the marker it is meant to circle -- node sizes themselves run 3..22px
var RING_EXTRA_PX = 10;

// global node indices of the current query's hits, INCLUDING ones currently
// hidden by a legend/dataset filter: they are counted separately so the UI
// can say a match exists but is filtered out, which is otherwise
// indistinguishable from no match at all.
var searchMatches = [];

// "?tag" tokens the last query used that resolve to no field, reported in the
// counter so a typo doesn't look like a graph with no matching nodes
var searchUnknownTags = [];

// whether the last query OR'd several unquoted words, which is worth pointing
// out if it then found nothing -- almost always an unquoted phrase
var searchPhraseHint = false;

// --- controls panel, bottom-right of the viewport and so outside the plot
// --- area (the nsewdrag rect) -- roughly under the figure's legend, which
// --- starts at the top of the right-hand margin. Fixed to the viewport
// --- rather than absolute inside gd's parent, matching what the NAFLD
// --- viewer's own panel already does.
var searchBox = document.createElement('div');

var PANEL_CSS = 'position:fixed; bottom:10px; right:10px; z-index:1000; background:white; ' +
    'border:1px solid #ccc; border-radius:6px; padding:10px 14px; font-family:sans-serif; ' +
    'font-size:12px; max-height:60vh; overflow-y:auto; box-shadow:0 1px 4px rgba(0,0,0,0.2);';

// With a host panel, searchBox is just a section inside it, prepended once
// filled so the edge toggle and the search sit ABOVE the host's own
// color-mode and dataset controls. Without one, searchBox IS the panel and
// carries the floating-box styling itself.
var hostPanel = (typeof controlPanel !== 'undefined') ? controlPanel : null;
if (!hostPanel) {
    searchBox.style.cssText = PANEL_CSS;
} else {
    // a separator, since the host's own title follows immediately below
    searchBox.style.cssText = 'border-bottom:1px solid #eee; margin-bottom:10px; padding-bottom:4px;';
}

var edgeTitle = document.createElement('div');
edgeTitle.textContent = 'Edges';
edgeTitle.style.cssText = 'font-weight:bold; margin-bottom:4px;';
searchBox.appendChild(edgeTitle);

var edgeLabel = document.createElement('label');
edgeLabel.style.display = 'block';
var edgeToggle = document.createElement('input');
edgeToggle.type = 'checkbox';
edgeToggle.checked = edgesVisible;
edgeLabel.appendChild(edgeToggle);
edgeLabel.appendChild(document.createTextNode(
    ' show edges (' + edgeU.length.toLocaleString() + ')'));
searchBox.appendChild(edgeLabel);

var edgeHint = document.createElement('div');
edgeHint.textContent = 'off by default: too dense to read unfiltered';
edgeHint.style.cssText = 'color:#888; margin:3px 0 10px;';
searchBox.appendChild(edgeHint);

var searchTitle = document.createElement('div');
searchTitle.textContent = 'Highlight nodes';
searchTitle.style.cssText = 'font-weight:bold; margin-bottom:4px;';
searchBox.appendChild(searchTitle);

var searchInput = document.createElement('input');
searchInput.type = 'search';
searchInput.placeholder = 'glucose   or   ?name "car 18:0"';
searchInput.title = 'Substring, case-insensitive, over every field. Prefix with ' +
    '?field to scope it:\n' +
    SEARCH_FIELD_TAGS.map(function (t) { return '?' + t; }).join('  ') +
    '\n\nSpace-separated terms are OR\'d, so quote anything containing a ' +
    'space: ?name "car 18:0"\nSeveral ?field groups can be combined ' +
    '("?name glucose ?formula c6h12o6").';
searchInput.style.cssText = 'width:230px; font-size:12px; padding:3px 5px;';
searchBox.appendChild(searchInput);

var searchHint = document.createElement('div');
searchHint.textContent = 'any field; space = OR, "quote a phrase"';
searchHint.style.cssText = 'color:#888; margin-top:3px;';
searchBox.appendChild(searchHint);

// the ?field vocabulary, collapsed by default -- twelve tags is too much to
// leave permanently on top of the plot, but guessing them is worse
var fieldDetails = document.createElement('details');
fieldDetails.style.cssText = 'color:#888; margin-top:2px;';
var fieldSummary = document.createElement('summary');
fieldSummary.textContent = 'scope to one field: ?field term';
fieldSummary.style.cursor = 'pointer';
fieldDetails.appendChild(fieldSummary);
var fieldList = document.createElement('div');
fieldList.textContent = SEARCH_FIELD_TAGS.map(function (t) { return '?' + t; }).join('  ');
fieldList.style.cssText = 'width:230px; margin-top:3px; line-height:1.5;';
fieldDetails.appendChild(fieldList);
searchBox.appendChild(fieldDetails);

var searchActions = document.createElement('div');
searchActions.style.marginTop = '4px';
var zoomLink = document.createElement('a');
zoomLink.textContent = 'zoom to matches';
zoomLink.href = '#';
zoomLink.style.marginRight = '10px';
var resetLink = document.createElement('a');
resetLink.textContent = 'reset view';
resetLink.href = '#';
searchActions.appendChild(zoomLink);
searchActions.appendChild(resetLink);
searchBox.appendChild(searchActions);

var searchCounter = document.createElement('div');
searchCounter.style.cssText = 'margin-top:4px; color:#444; min-height:1em;';
searchBox.appendChild(searchCounter);

if (hostPanel) {
    hostPanel.insertBefore(searchBox, hostPanel.firstChild);
} else {
    document.body.appendChild(searchBox);
}

// Split a query into terms, keeping quoted runs whole so a term may contain
// spaces. The closing quote is optional -- while the user is still typing
// `?name "car 18` the opening quote should start a phrase that runs to the
// end, not leave a stray `"car` token behind.
//
// Every branch consumes at least one character (the quote, or one non-space),
// so exec() can never match empty and spin forever.
var TOKEN_RE = /"([^"]*)"?|'([^']*)'?|([^\s,;]+)/g;

function tokenize(query) {
    var tokens = [];
    var match;
    TOKEN_RE.lastIndex = 0;
    while ((match = TOKEN_RE.exec(query)) !== null) {
        var quoted = match[1] !== undefined || match[2] !== undefined;
        var text = quoted ? (match[1] !== undefined ? match[1] : match[2]) : match[3];
        if (text.length > 0) tokens.push({text: text, quoted: quoted});
    }
    return tokens;
}

// Split the tokens into {field, terms} groups: field -1 means "any field",
// and a group is dropped if its tag was unrecognized or it ended up with no
// terms (e.g. a trailing "?formula" the user hasn't finished typing).
function parseQuery(query) {
    var tokens = tokenize(query.toLowerCase());
    var unknown = [];
    var groups = [];
    var current = {field: -1, terms: []};
    var multiTermGroup = false;

    for (var k = 0; k < tokens.length; k++) {
        var token = tokens[k];
        // a quoted token is always a term, never a tag, so a name that really
        // does start with "?" can still be searched for
        if (token.quoted || token.text.charAt(0) !== '?') {
            current.terms.push(token.text);
            if (current.terms.length > 1 && !token.quoted) multiTermGroup = true;
            continue;
        }
        groups.push(current);
        var tag = token.text.slice(1);
        if (Object.prototype.hasOwnProperty.call(SEARCH_FIELD_INDEX, tag)) {
            current = {field: SEARCH_FIELD_INDEX[tag], terms: []};
        } else {
            unknown.push(token.text);
            current = {field: null, terms: []};   // collects, then gets dropped
        }
    }
    groups.push(current);

    return {
        groups: groups.filter(function (g) { return g.field !== null && g.terms.length > 0; }),
        unknown: unknown,
        // used only to suggest quoting when such a query finds nothing, which
        // is the signature of a phrase that got split into OR'd words
        multiTermGroup: multiTermGroup,
    };
}

function nodeMatches(fields, groups) {
    for (var g = 0; g < groups.length; g++) {
        var group = groups[g];
        for (var t = 0; t < group.terms.length; t++) {
            var term = group.terms[t];
            if (group.field >= 0) {
                if (fields[group.field].indexOf(term) !== -1) return true;
            } else {
                for (var f = 0; f < fields.length; f++) {
                    if (fields[f].indexOf(term) !== -1) return true;
                }
            }
        }
    }
    return false;
}

function runSearch(query) {
    var parsed = parseQuery(query);
    searchUnknownTags = parsed.unknown;
    searchPhraseHint = parsed.multiTermGroup;
    searchMatches = [];
    if (parsed.groups.length === 0) return;
    for (var i = 0; i < N; i++) {
        if (nodeMatches(nodeSearchFields[i], parsed.groups)) searchMatches.push(i);
    }
}

function applySearchHighlight() {
    var x = [], y = [], size = [], hidden = 0;
    for (var k = 0; k < searchMatches.length; k++) {
        var i = searchMatches[k];
        if (!nodeVisible(i)) {
            hidden++;
            continue;
        }
        x.push(nodeX[i]);
        y.push(nodeY[i]);
        size.push(nodeSize[i] + RING_EXTRA_PX);
    }
    Plotly.restyle(gd, {x: [x], y: [y], 'marker.size': [size]}, [HIGHLIGHT_TRACE_INDEX]);
    return {shown: x.length, hidden: hidden};
}

function refreshSearch() {
    var counts = applySearchHighlight();
    var parts = [];
    if (searchUnknownTags.length) {
        parts.push('unknown field ' + searchUnknownTags.join(', '));
    }
    if (searchInput.value.trim().length === 0) {
        // nothing typed: stay quiet rather than reporting "no match"
    } else if (counts.shown === 0 && counts.hidden === 0) {
        parts.push('no match');
        if (searchPhraseHint) parts.push('quote a phrase: "organic acids"');
    } else {
        parts.push(counts.shown + ' highlighted' +
            (counts.hidden ? ' (' + counts.hidden + ' matched but hidden by filters)' : ''));
    }
    searchCounter.textContent = parts.join(' -- ');
}

function zoomToMatches() {
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, n = 0;
    for (var k = 0; k < searchMatches.length; k++) {
        var i = searchMatches[k];
        if (!nodeVisible(i)) continue;
        // accumulated in a loop rather than via Math.min.apply(null, xs):
        // a broad query can match thousands of nodes, and apply() with that
        // many arguments overflows the call stack
        if (nodeX[i] < x0) x0 = nodeX[i];
        if (nodeX[i] > x1) x1 = nodeX[i];
        if (nodeY[i] < y0) y0 = nodeY[i];
        if (nodeY[i] > y1) y1 = nodeY[i];
        n++;
    }
    if (n === 0) return;
    // a single match has a zero-size bounding box, which would collapse both
    // axes to a point -- fall back to a fixed window around it instead
    var padX = (x1 - x0) * 0.15 || 5;
    var padY = (y1 - y0) * 0.15 || 5;
    Plotly.relayout(gd, {
        'xaxis.range': [x0 - padX, x1 + padX],
        'yaxis.range': [y0 - padY, y1 + padY],
    });
}

searchInput.addEventListener('input', function () {
    runSearch(searchInput.value);
    refreshSearch();
});

zoomLink.onclick = function (e) {
    e.preventDefault();
    zoomToMatches();
};

resetLink.onclick = function (e) {
    e.preventDefault();
    Plotly.relayout(gd, {'xaxis.autorange': true, 'yaxis.autorange': true});
};

// The host script calls recomputeEdges() after every visibility change, so
// rebinding it is the one place that can both gate the edge rebuild on the
// toggle and keep the rings (and the "hidden by filters" count) in step,
// without having to touch each of the host's call sites. Rebinding a function
// *declaration* works because the two scripts share one script scope, and the
// host calls it by name -- so its call sites pick up this wrapper.
var hostRecomputeEdges = recomputeEdges;
recomputeEdges = function () {
    if (edgesVisible) {
        hostRecomputeEdges();
    } else {
        // blank both edge traces rather than setting visible:false, so the
        // host's own restyles (which only ever write x/y) can't bring them
        // back while the toggle is off
        Plotly.restyle(gd, {x: [[], []], y: [[], []]}, [0, 1]);
    }
    refreshSearch();
};

edgeToggle.onchange = function () {
    edgesVisible = edgeToggle.checked;
    recomputeEdges();
};
