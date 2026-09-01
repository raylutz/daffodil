/*
daffodil-csv.js -- read a Daf.to_csv_buff()-produced CSV into plain JS objects, browser-side, with
no build step and no server.

Mirrors the read path Daf.from_csv_buff() -> apply_dtypes() -> convert_type_value() ->
unflatten_val() uses in src/daffodil/lib/daf_utils.py: any cell shaped like [...], {...}, or (...)
is a flattened list/dict/tuple (Daf.to_csv_buff() calls this "PYON" -- like JSON but with
single-quoted strings allowed, non-string dict keys allowed, and True/False instead of true/false)
and gets unflattened back to a real object. This is shape-driven, not schema-driven -- no dtypes
declaration is required, matching the Python side's own behavior.

JSON5 (vendored in vendor/json5.js, load it first) already accepts PYON's single-quoted
strings/dicts/lists directly. The one real gap is Python's capitalized True/False/None, which
JSON5 does not accept as bare keywords -- pyonKeywordsToJson5() rewrites just those tokens,
skipping over quoted string content so a value like 'None of the above' is left untouched.

Depends on the global JSON5 (see vendor/json5.js) -- load that first:
    <script src="vendor/json5.js"></script>
    <script src="daffodil-csv.js"></script>
*/
var Daffodil = (function() {
  "use strict";

  // Minimal RFC4180 CSV parser: handles quoted fields containing commas, quotes ("" escaping),
  // and embedded newlines. This is the only CSV parsing this file does -- PYON/JSON5 decoding of
  // individual cell values is a separate, later step (unflattenCell()).
  function parseCsv(text) {
    var rows = [];
    var row = [];
    var field = '';
    var inQuotes = false;
    for (var i = 0; i < text.length; i++) {
      var c = text[i];
      if (inQuotes) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; }
          else { inQuotes = false; }
        } else {
          field += c;
        }
      } else if (c === '"') {
        inQuotes = true;
      } else if (c === ',') {
        row.push(field); field = '';
      } else if (c === '\r') {
        // skip, handled by the following \n
      } else if (c === '\n') {
        row.push(field); rows.push(row); row = []; field = '';
      } else {
        field += c;
      }
    }
    if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
    return rows;
  }

  function csvRowsToObjects(rows) {
    if (rows.length === 0) return [];
    var header = rows[0];
    var out = [];
    for (var r = 1; r < rows.length; r++) {
      if (rows[r].length === 1 && rows[r][0] === '') continue;   // trailing blank line
      var obj = {};
      for (var c = 0; c < header.length; c++) { obj[header[c]] = rows[r][c]; }
      out.push(obj);
    }
    return out;
  }

  // Rewrite bare Python True/False/None keywords to JSON5-legal true/false/null, leaving quoted
  // string content untouched (so a candidate/write-in value like 'None of the above' or a string
  // that happens to contain the word True is not corrupted). Tracks single- and double-quote
  // state and backslash escapes the same way parseCsv() and JSON5 itself do.
  var PYON_KEYWORD_RE = /^(True|False|None)\b/;
  var PYON_KEYWORD_JSON5 = {True: 'true', False: 'false', None: 'null'};

  function pyonKeywordsToJson5(s) {
    var out = '';
    var i = 0;
    var quote = null;
    while (i < s.length) {
      var c = s[i];
      if (quote) {
        out += c;
        if (c === '\\') { out += s[i + 1] || ''; i += 2; continue; }
        if (c === quote) { quote = null; }
        i++; continue;
      }
      if (c === "'" || c === '"') { quote = c; out += c; i++; continue; }
      var m = PYON_KEYWORD_RE.exec(s.slice(i));
      if (m) { out += PYON_KEYWORD_JSON5[m[1]]; i += m[1].length; continue; }
      out += c; i++;
    }
    return out;
  }

  // Python tuple repr, e.g. "(1, 2, 3)", isn't valid JSON5 (no tuple syntax) -- treat it as a
  // list, same as JSON has no tuple type either. Only rewrites the outer parens, not any nested
  // ones inside string content, since quote state is tracked the same way as above.
  function tupleParensToBrackets(s) {
    if (s[0] === '(' && s[s.length - 1] === ')') {
      return '[' + s.slice(1, -1) + ']';
    }
    return s;
  }

  // Unflatten one CSV cell value if it looks like a flattened list/dict/tuple, else return it
  // unchanged as a plain string. Falls back to the original string (not a thrown error) if it
  // looks structured but doesn't actually parse, matching unflatten_val()'s own fallback.
  function unflattenCell(val) {
    if (typeof val !== 'string') return val;
    var trimmed = val.trim();
    if (trimmed.length < 2) return val;

    var first = trimmed[0];
    var last = trimmed[trimmed.length - 1];
    var looksStructured =
      (first === '{' && last === '}') ||
      (first === '[' && last === ']') ||
      (first === '(' && last === ')');
    if (!looksStructured) return val;

    var json5Text = pyonKeywordsToJson5(tupleParensToBrackets(trimmed));
    try {
      return JSON5.parse(json5Text);
    } catch (e) {
      return val;
    }
  }

  // Parse a full CSV buffer into an array of plain objects, one per row, keyed by the header row.
  // unflatten (default true, matching Daf.from_csv_buff()'s own default) controls whether
  // list/dict/tuple-shaped cells get decoded into real arrays/objects or left as raw strings.
  function csvToObjects(text, opts) {
    opts = opts || {};
    var unflatten = opts.unflatten !== false;
    var rows = csvRowsToObjects(parseCsv(text));
    if (!unflatten) return rows;

    return rows.map(function(row) {
      var out = {};
      Object.keys(row).forEach(function(key) { out[key] = unflattenCell(row[key]); });
      return out;
      });
  }

  return {
    csvToObjects:        csvToObjects,
    parseCsv:             parseCsv,
    csvRowsToObjects:     csvRowsToObjects,
    unflattenCell:        unflattenCell,
    pyonKeywordsToJson5:  pyonKeywordsToJson5,
    };
})();
