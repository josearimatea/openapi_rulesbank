# src/prompts/rule_type_definitions.py

"""
The one definition of each rule type, handed to every stage that decides or judges
one (Planner, section analyzer, scope, Extractor, Reflector, Validator).

Each stage used to carry its own wording of what a path, a parameter or a callback
is, and the wordings drifted apart: one prompt said a resource on the consumer is
always a callback while another said a fixed path is a path_operation, and each
stage acted on its own copy. Here every definition is written once, in two
labelled parts, so a later reader can tell where each statement comes from:

  REFERENCE        — quoted from the OpenAPI Specification 3.0.1 (the version of
                     the 3GPP YAMLs), with the object each quote comes from.
  3GPP CONVENTION  — how 3GPP specifications and the published YAMLs express it.
                     Observed, NOT in the reference; each says where it was
                     verified, so it can be re-checked when a new spec disagrees.

Only the pipeline's own rule contract (openapi_object / openapi_field /
openapi_value formats) and its procedures stay in the stage prompts; what a
construct IS lives only here.

The same vocabulary also drives section_signals(), the per-section hints the
Planner reads, so the definition a stage reads and the signal the code finds
cannot disagree.
"""

import re

_HEADER = (
    "RULE TYPE DEFINITIONS — the one definition every stage uses. REFERENCE is "
    "quoted from the OpenAPI Specification 3.0.1. 3GPP CONVENTION is how 3GPP "
    "specifications and the published YAMLs express it: observed, not in the "
    "reference."
)

RULE_TYPE_DEFINITIONS: dict[str, str] = {

"path_operation": """\
path_operation
  REFERENCE (OpenAPI 3.0.1):
    - Paths Object: "Holds the relative paths to the individual endpoints and
      their operations. The path is appended to the URL from the Server Object in
      order to construct the full URL."
    - Paths Object, /{path}: "A relative path to an individual endpoint. The field
      name MUST begin with a slash. The path is appended (no relative URL
      resolution) to the expanded URL from the Server Object's url field in order
      to construct the full URL. Path templating is allowed."
    - Path Item Object: "Describes the operations available on a single path."
      Its operation fields are get, put, post, delete, options, head, patch,
      trace — each "A definition of a <METHOD> operation on this path."
    - Path Templating: "Path templating refers to the usage of curly braces ({})
      to mark a section of a URL path as replaceable using path parameters."
    - Operation Object: "Describes a single API operation on a path."
    - Server Object, url: "A URL to the target host. This URL supports Server
      Variables ... Variable substitutions will be made when a variable is named
      in {brackets}."
  3GPP CONVENTION (observed, not in the reference):
    - A spec's resource URI is the API root followed by the resource. The API root
      is the part the spec defines as common to every resource of the API — a
      root variable, the API or service name, its version, and any other
      variable the spec defines by pointing at a common URI structure rather
      than at this resource. It goes under `servers` (url and variables); only
      the resource part, from its first "/", is the path key. [Verified: TS 28.532
      ProvMnS YAML — URI {MnSRoot}/ProvMnS/{MnSVersion}/{URI-LDN-first-part}/
      {className}={id}: servers url '{MnSRoot}/ProvMnS/{MnSVersion}/
      {URI-LDN-first-part}', path '/{className}={id}'.]
    - The path key is the resource the spec gives, never a name built from an
      operation or a notification; the operation is the HTTP method on it.
      [Verified: TS 28.532 YAMLs — no path is named after an IS operation or
      notification.]
    - Who hosts a resource does not decide whether it is a path. A resource the
      spec places on the consumer side that has a CONCRETE path is a path of the
      API that describes that resource; what a subscription or registration
      supplies there is the server root, not the path. [Verified: TS 28.532
      PerfMnS — table 12.3.1.2.2.1-1 maps notifyThresholdCrossing to POST
      /notificationSink; its YAML has paths: /notificationSink and a server root
      "located in the consumer side".]""",

"path_parameter": """\
path_parameter
  REFERENCE (OpenAPI 3.0.1):
    - Parameter Object: "Describes a single operation parameter. A unique
      parameter is defined by a combination of a name and location."
    - Path Templating: "Path templating refers to the usage of curly braces ({})
      to mark a section of a URL path as replaceable using path parameters."
    - Parameter Object, in: path: "Used together with Path Templating, where the
      parameter value is actually part of the operation's URL. This does not
      include the host or base path of the API. For example, in /items/{itemId},
      the path parameter is itemId."
    - Parameter Object, required: "If the parameter location is "path", this
      property is REQUIRED and its value MUST be true."
    - Path Item Object, parameters: "A list of parameters that are applicable for
      all the operations described under this path. These parameters can be
      overridden at the operation level, but cannot be removed there."
    - Operation Object, parameters: "A list of parameters that are applicable for
      this operation. If a parameter is already defined at the Path Item, the new
      definition will override it but can never remove it."
  3GPP CONVENTION (observed, not in the reference):
    - A URI variable of the API root (see path_operation: defined as common to
      every resource of the API, often by pointing at a common URI structure
      specification) is a server variable, not a path parameter. Only a variable
      inside the resource part of the URI is a path_parameter. [Verified: TS
      28.532 ProvMnS YAML — MnSRoot, MnSVersion and URI-LDN-first-part, defined by
      pointing at TS 32.158, are server variables; className and id are path
      parameters.]""",

"query_parameter": """\
query_parameter
  REFERENCE (OpenAPI 3.0.1):
    - Parameter Object: "Describes a single operation parameter. A unique
      parameter is defined by a combination of a name and location."
    - Parameter Object, in: query: "Parameters that are appended to the URL. For
      example, in /items?id=###, the query parameter is id."
    - Operation Object, parameters: "A list of parameters that are applicable for
      this operation."
    - Path Item Object, parameters: "A list of parameters that are applicable for
      all the operations described under this path."
  3GPP CONVENTION (observed, not in the reference):
    - The published YAMLs declare query parameters on the operation that accepts
      them, not on the path item. [Verified: the six TS 28.532 YAMLs in
      data/inputs/yamls.]""",

"request_body": """\
request_body
  REFERENCE (OpenAPI 3.0.1):
    - Operation Object, requestBody: "The request body applicable for this
      operation. The requestBody is only supported in HTTP methods where the HTTP
      1.1 specification RFC7231 has explicitly defined semantics for request
      bodies."
    - Request Body Object: "Describes a single request body."
    - Request Body Object, content: "REQUIRED. The content of the request body.
      The key is a media type or media type range and the value describes it."
    - Media Type Object: "Each Media Type Object provides schema and examples
      for the media type identified by its key."
    - Media Type Object, schema: "The schema defining the type used for the
      request body."
    - Media Types: "The media type definitions SHOULD be in compliance with
      RFC6838." Examples given: "text/plain; charset=utf-8", "application/json",
      "application/vnd.github+json".
  3GPP CONVENTION (observed, not in the reference):
    - A clause naming only a data type is carried in application/json.
      [Verified: request bodies of the TS 28.532 YAMLs.]
    - A body's schema is the data type the spec says the body carries, written
      as a $ref to that type (never an inline copy). The spec may say it in a
      different clause from the one that says a body exists: one clause states
      that an operation's or notification's parameters travel "in the request
      body", another — typically the clause of the HTTP method on the resource —
      names the data type of that body, sometimes one type per operation or
      notification it serves. Both describe the SAME body: one request_body,
      whose schema is that type. [Verified: TS 28.532 — the notification clauses
      (PerfMnS 12.3.1.2.2.2; ProvMnS 12.1.1.2.2-2.5) map parameters to "request
      body" without naming the type; the POST clauses (12.3.1.2.3.2.1.3.1;
      12.1.1.3.2.4.3.1) name it in a "Data structures supported by the POST
      Request Body" table; every requestBody of the TS 28.532 YAMLs has
      schema $ref to that type.]""",

"response": """\
response
  REFERENCE (OpenAPI 3.0.1):
    - Operation Object, responses: "REQUIRED. The list of possible responses as
      they are returned from executing this operation."
    - Responses Object: "A container for the expected responses of an operation.
      The container maps a HTTP response code to the expected response. ...
      documentation is expected to cover a successful operation response and any
      known errors. The default MAY be used as a default response object for all
      HTTP codes that are not covered individually by the specification."
    - Responses Object, HTTP Status Code: "Any HTTP status code can be used as the
      property name, but only one property per code ... To define a range of
      response codes, this field MAY contain the uppercase wildcard character X.
      For example, 2XX represents all response codes between [200-299]."
    - Responses Object, default: "The documentation of responses other than the
      ones declared for specific HTTP response codes."
    - Response Object: "Describes a single response from an API Operation."
    - Response Object, content: "A map containing descriptions of potential
      response payloads. The key is a media type or media type range and the
      value describes it."
    - Media Type Object: "Each Media Type Object provides schema and examples
      for the media type identified by its key."
    - Response Object, headers: "Maps a header name to its definition."
    - HTTP Status Codes: "The HTTP Status Codes are used to indicate the status
      of the executed operation. The available status codes are defined by
      RFC7231 and registered status codes are listed in the IANA Status Code
      Registry."
    - Media Types: "The media type definitions SHOULD be in compliance with
      RFC6838."
  3GPP CONVENTION (observed, not in the reference): none recorded.""",

"schema_property": """\
schema_property
  REFERENCE (OpenAPI 3.0.1):
    - Schema Object: "The Schema Object allows the definition of input and output
      data types. These types can be objects, but also primitives and arrays."
    - Components Object, schemas: "An object to hold reusable Schema Objects."
    - Data Types: "Primitive data types in the OAS are based on the types
      supported by the JSON Schema Specification Wright Draft 00. Note that
      integer as a type is also supported ... Primitives have an optional
      modifier property: format." The formats it defines: integer (int32,
      int64), number (float, double), string (byte, binary, date, date-time,
      password), boolean.
  3GPP CONVENTION (observed, not in the reference):
    - An attribute whose type is a named type defined elsewhere is a $ref to that
      schema, never an inline copy. [Verified: TS 28.532 YAMLs.]
    - A type built on a base type is allOf: [$ref to the base, its own
      properties]; the base's fields are not repeated. [Verified: TS 28.532
      PerfMnS YAML — NotifyThresholdCrossing is allOf NotificationHeader.]""",

"security_scheme": """\
security_scheme
  REFERENCE (OpenAPI 3.0.1):
    - Security Scheme Object: "Defines a security scheme that can be used by the
      operations. Supported schemes are HTTP authentication, an API key (either
      as a header or as a query parameter), OAuth2's common flows ... and OpenID
      Connect Discovery."
    - Security Scheme Object, type: "REQUIRED. The type of the security scheme.
      Valid values are "apiKey", "http", "oauth2", "openIdConnect"."
    - Security Scheme Object, in (apiKey): "The location of the API key. Valid
      values are "query", "header" or "cookie"."
    - Security Scheme Object, scheme (http): "The name of the HTTP Authorization
      scheme to be used in the Authorization header as defined in RFC7235."
  3GPP CONVENTION (observed, not in the reference): none recorded.""",

"callback": """\
callback
  REFERENCE (OpenAPI 3.0.1):
    - Operation Object, callbacks: "A map of possible out-of band callbacks
      related to the parent operation. ... Each value in the map is a Callback
      Object that describes a request that may be initiated by the API provider
      and the expected responses. The key value used to identify the callback
      object is an expression, evaluated at runtime, that identifies a URL to use
      for the callback operation."
    - Callback Object, Key Expression: "The key that identifies the Path Item
      Object is a runtime expression that can be evaluated in the context of a
      runtime HTTP request/response to identify the URL to be used for the
      callback request. A simple example might be $request.body#/url."
    - Runtime Expressions: "Runtime expressions allow defining values based on
      information that will only be available within the HTTP message in an
      actual API call." Syntax: expression = ( "$url" | "$method" |
      "$statusCode" | "$request." source | "$response." source ); source = (
      header-reference | query-reference | path-reference | body-reference );
      body-reference = "body" ["#" fragment], the fragment a JSON Pointer.
  3GPP CONVENTION (observed, not in the reference):
    - The delivery address reaches the provider in the request of one of THIS
      API's operations — usually a subscription: an object the consumer creates
      through the API's create operation, or a dedicated subscribe operation. The
      callback hangs off that operation. A service with no operation of its own
      that carries the address has no callbacks: what it describes for delivery
      are the receiver's paths. [Verified: TS 28.532 — ProvMnS (address in the
      subscription created by PUT) and FileDataReporting (POST /subscriptions)
      YAMLs have callbacks; PerfMnS and Heartbeat, whose operation-mapping
      clauses are empty, have none.]
    - Where a spec maps a notification to an HTTP method and a URI, the
      consumer's runtime address is written as a URI placeholder (a variable in
      braces standing for the address), while a resource of an API is written as
      a concrete path. The placeholder is a clue for the test above, not a test
      of its own. [Verified: TS 28.532 notification-mapping tables 12.1.1.2.1-1
      and 12.6.1.2.1-1 ({notificationTarget} -> callbacks in the ProvMnS and
      FileDataReporting YAMLs); 12.3.1.2.2.1-1 and 12.4.1.2.1-1
      (/notificationSink -> no callbacks in the PerfMnS and HeartbeatNtf YAMLs).]
    - The published YAMLs write the key as '{request.body#/<field>}', without the
      '$' of the reference's examples ('{$request.body#/id}'). [Verified: no
      '$request' in data/inputs/yamls.]""",
}


def format_rule_type_definitions(rule_types: list[str] | None = None) -> str:
    """The definitions block, for all rule types or only the ones named."""
    chosen = rule_types or list(RULE_TYPE_DEFINITIONS)
    return _HEADER + "\n\n" + "\n\n".join(
        RULE_TYPE_DEFINITIONS[t] for t in chosen if t in RULE_TYPE_DEFINITIONS
    )


def rule_type_definitions_for_template(rule_types: list[str] | None = None) -> str:
    """The same block with braces doubled, to sit inside a ChatPromptTemplate."""
    return format_rule_type_definitions(rule_types).replace("{", "{{").replace("}", "}}")


# ---------------------------------------------------------------------------
# The same vocabulary as patterns — what a section's text shows of each rule
# type, read by code over the WHOLE section (the Planner sees only its start).
# Every pattern is taken from a REFERENCE entry above, named in its comment; a
# signal is a hint for the Planner, never a decision.
#
# !! DELICATE — FALSE POSITIVES !!
# Pattern matching cannot tell the sense a word is used in. Cases met while
# building this (28/out/2026, TS 28.532) and the narrowing each one got:
#   - a JSON Pointer in an example ("/attributes/attrA") or a closing XML tag
#     ("</element>") read as a path        -> paths only on a method/URI/resource line
#   - an example value ("123") read as a status code
#                                            -> codes only where a response/status
#                                               is spoken of (line or table caption)
#   - "media/image4.emf", "name/value" read as media types
#                                            -> RFC6838 top-level types only;
#                                               image lines skipped
#   - "{notificationTarget}" (an address) read as a path parameter
#                                            -> a bare {var} URI goes under
#                                               "uri placeholder", no rule type
# Other specs may bring cases not seen here. Switch: settings.PLANNER_SECTION_
# SIGNALS. If the Planner is misled by a wrong signal, turn it off and record the
# case here before narrowing a pattern further.
# ---------------------------------------------------------------------------

# Path Item Object: its eight operation fields. Upper case, as HTTP writes them.
_METHOD = re.compile(r"\b(GET|PUT|POST|DELETE|OPTIONS|HEAD|PATCH|TRACE)\b")
# Paths Object: "The field name MUST begin with a slash"; Path Templating: {name}.
# Taken only on a line that names a method or a URI/resource: elsewhere a slash
# path is a JSON Pointer in an example or a closing XML tag. On such a line a URI
# that is a template variable alone ({name}) is taken too — the spec's way of
# writing an address it does not fix.
_PATH = re.compile(r"(?<![\w/.:<\-])(/[A-Za-z{.][\w{}=.\-]*(?:/[\w{}=.\-]+)*|\{[A-Za-z][\w\-]*\})")
_PATH_CONTEXT = re.compile(r"\b(GET|PUT|POST|DELETE|OPTIONS|HEAD|PATCH|TRACE)\b|\bURI\b|\bresource\b", re.I)
_TEMPLATE_VAR = re.compile(r"\{([A-Za-z][\w\-]*)\}")
# Parameter Object, in: query — "Parameters that are appended to the URL ...
# /items?id=###".
_QUERY = re.compile(r"\?[A-Za-z][\w\-]*=|\bquery parameters?\b", re.I)
# Responses Object: an HTTP status code or a range 1XX..5XX. A number is taken
# only where a response or status is spoken of (the Responses Object's own
# terms): on such a line, or on a row of a table whose caption says so — so
# clause numbers and example values are not read as codes.
_STATUS = re.compile(r"\b([1-5][0-9]{2}|[1-5]XX)\b", re.I)
_STATUS_CONTEXT = re.compile(r"respon|status", re.I)
# Media Types: "SHOULD be in compliance with RFC6838" — type/subtype, the form of
# the reference's examples (text/plain, application/json,
# application/vnd.github+json). The type is one of the top-level types RFC6838
# registers, which the reference requires compliance with.
_MEDIA = re.compile(r"(?<![\w./:\-])((?:application|audio|example|font|image|message|"
                    r"model|multipart|text|video)/[a-z0-9][a-z0-9.+\-]*)\b")
_CAPTION = re.compile(r"^\**Table\s+[0-9A-Z][\w.\-]*:", re.I)
# Operation Object, requestBody.
_REQUEST_BODY = re.compile(r"\brequest[ -]?body\b", re.I)
# Data Types: the types and formats the reference lists, plus array and object
# (Schema Object items / properties). Taken only on table rows, where a data-type
# column holds them; in prose "string" or "object" are ordinary words.
_DATA_TYPE = re.compile(r"\b(integer|int32|int64|number|float|double|string|byte|"
                        r"binary|boolean|date-time|date|password|array|object)\b")
# Security Scheme Object: type values, and the HTTP Authorization header (scheme).
_SECURITY = re.compile(r"\b(apiKey|oauth2|OAuth|openIdConnect|OpenID|Authorization header)\b")
# Runtime Expressions: "$request." / "$response." + header/query/path/body, and
# $url, $method, $statusCode — with or without the '$' (3GPP CONVENTION above).
_RUNTIME = re.compile(r"\$?(?:request|response)\.(?:body|header|query|path)\b[#\w/.\-]*"
                      r"|\$(?:url|method|statusCode)\b")


def _is_table_row(line: str) -> bool:
    """A grid-table row ('|' cells) or a simple-table row (columns apart by 2+ spaces)."""
    return "|" in line or bool(re.search(r"\S\s{2,}\S.*\S\s{2,}\S", line))


def section_signals(content: str, limit: int = 6) -> dict[str, list[str]]:
    """What the section's whole text shows of each rule type, by rule type.

    Distinct values in order of appearance, at most `limit` per type; a type the
    text shows nothing of is left out.
    """
    found: dict[str, list[str]] = {}

    def add(rule_type: str, value: str) -> None:
        values = found.setdefault(rule_type, [])
        if value not in values and len(values) < limit:
            values.append(value)

    # The caption of the table the current line belongs to. A caption may wrap
    # over several lines up to a blank one, and holds until the next caption.
    caption, in_caption = "", False
    for line in content.splitlines():
        if line.lstrip().startswith("!["):
            continue  # an image reference, not text
        # Caption lines are read too: a method clause may name its method and its
        # request body only there ("... supported by the POST Request Body").
        if _CAPTION.match(line.strip()):
            caption, in_caption = line, True
        elif in_caption and line.strip():
            caption += " " + line
        else:
            in_caption = False
        for m in _METHOD.finditer(line):
            add("path_operation", m.group(1))
        if _PATH_CONTEXT.search(line):
            for m in _PATH.finditer(line):
                path = m.group(1).rstrip(".")
                if not path.startswith("/"):
                    # A URI that is a template variable alone: what it stands for
                    # is for the definitions to settle (callback / path), so it is
                    # reported as it is, under no rule type.
                    add("uri placeholder", path)
                    continue
                add("path_operation", path)
                for v in _TEMPLATE_VAR.finditer(path):
                    add("path_parameter", "{" + v.group(1) + "}")
        if _QUERY.search(line):
            add("query_parameter", "query parameters")
        if _STATUS_CONTEXT.search(line) or (_is_table_row(line) and _STATUS_CONTEXT.search(caption)):
            for m in _STATUS.finditer(line):
                add("response", m.group(1).upper())
        for m in _MEDIA.finditer(line):
            add("request_body", m.group(1))
        if _REQUEST_BODY.search(line):
            add("request_body", "request body")
        if _is_table_row(line):
            for m in _DATA_TYPE.finditer(line):
                add("schema_property", m.group(1))
        for m in _SECURITY.finditer(line):
            add("security_scheme", m.group(1))
        for m in _RUNTIME.finditer(line):
            add("callback", m.group(0))
    return found


def format_section_signals(content: str) -> str:
    """section_signals on one line: 'type: v1, v2; type: v3', or 'none'."""
    found = section_signals(content)
    return "; ".join(f"{t}: {', '.join(v)}" for t, v in found.items()) or "none"
