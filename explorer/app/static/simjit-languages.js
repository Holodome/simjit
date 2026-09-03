// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

(() => {
  if (!window.hljs?.registerLanguage) return;

  const cppIntrinsicTypes = [
    "__m64",
    "__m128",
    "__m128d",
    "__m128i",
    "__m256",
    "__m256d",
    "__m256i",
    "__m512",
    "__m512d",
    "__m512i",
    "__mmask8",
    "__mmask16",
    "__mmask32",
    "__mmask64",
  ];

  function extendCppIntrinsicTypes() {
    const cpp = window.hljs.getLanguage?.("cpp");
    const types = cpp?.keywords?.type;
    if (!Array.isArray(types)) return;
    const existing = new Set(
      types.map((value) => String(value).split("|", 1)[0]),
    );
    cppIntrinsicTypes.forEach((type) => {
      if (!existing.has(type)) types.push(type);
    });
  }

  function extendCppUnderscoreDispatch() {
    const cpp = window.hljs.getLanguage?.("cpp");
    if (!cpp || cpp.__simjitUnderscoreDispatch) return;
    if (cpp.classNameAliases) {
      cpp.classNameAliases["function.dispatch"] = "title";
    }
    const mode = {
      scope: "title",
      match: /\b_+\w+(?=(?:<[^<>]+>)?\s*\()/,
      relevance: 0,
    };
    const visited = new Set();

    function install(container) {
      if (!container || typeof container !== "object" || visited.has(container)) {
        return;
      }
      visited.add(container);
      if (!Array.isArray(container.contains)) return;
      if (Object.isExtensible(container.contains)) {
        container.contains.unshift(mode);
      }
      container.contains.forEach(install);
    }

    install(cpp);
    cpp.__simjitUnderscoreDispatch = true;
  }

  const reference = { scope: "symbol", match: /[@$%]\d+\b/ };
  const dtype = {
    scope: "type",
    match:
      /\b(?:i1|i8|i16|i32|i64|u1|u8|u16|u32|u64|f16|f32|f64|m1|m2|m4|m8|m16|m32|m64|[iuf]\d+x\d+)\b/,
  };
  const number = {
    scope: "number",
    match: /\b-?(?:0x[0-9a-f]+|\d+(?:\.\d+)?)\b/i,
  };
  const operator = { scope: "operator", match: /<-|->|=>|[=:+\-*/<>!&|]+/ };
  const quoted = { scope: "string", match: /"[^"]*"|'[^']*'/ };

  const keyword = {
    scope: "keyword",
    match:
      /\b(?:acc-arith-bin|acc_arith_bin|accload|accstore|aggresult|acc(?=\s|$)|arg(?=\s|$)|binary|bitcast|call|cmp|const|extend|fp-cast|function|gather|index|int-cast|load|node|null|predicate-not|reduce|select|store|truncate|unary|predicate-binary|mask-binary|const-div|mask-count|countif)\b/,
  };
  const section = {
    scope: "section",
    match: /^(?:ACCUMS|EPILOGUE|MAIN LOOP|PROLOGUE|REMAINDER)$/m,
  };

  function simjitLanguage(name, extras = []) {
    return (hljs) => ({
      name,
      case_insensitive: false,
      contains: [
        hljs.COMMENT("#", "$"),
        section,
        reference,
        dtype,
        number,
        quoted,
        keyword,
        operator,
        ...extras,
      ],
    });
  }

  window.hljs.registerLanguage("simjit-hir", simjitLanguage("Simjit HIR"));
  window.hljs.registerLanguage(
    "simjit-vectorizer",
    simjitLanguage("Simjit Vectorizer", [
      { scope: "meta", match: /^\s*\[\d+\]/ },
      {
        scope: "title",
        match: /\b(?:success|failure|fallback|scalar|vector|native|x86|arm)\b/,
      },
    ]),
  );
  window.hljs.registerLanguage("simjit-mir", simjitLanguage("Simjit MIR"));
  extendCppIntrinsicTypes();
  extendCppUnderscoreDispatch();
})();
