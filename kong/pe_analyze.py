"""Fast PE analysis — extract useful intelligence without Ghidra or decompilation.

Works on packed binaries. Extracts exports, imports, version info, strings,
section layout, and packing indicators directly from the PE file.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PEExport:
    name: str
    ordinal: int
    rva: int


@dataclass
class PEImportDll:
    dll: str
    functions: list[str]


@dataclass
class PESectionDetail:
    name: str
    virtual_size: int
    raw_size: int
    entropy: float
    is_executable: bool
    is_empty: bool


@dataclass
class PEVersionInfo:
    company: str = ""
    description: str = ""
    version: str = ""
    product_name: str = ""
    copyright: str = ""
    original_filename: str = ""


@dataclass
class PEAnalysis:
    """Complete fast analysis of a PE file."""
    filename: str
    file_size: int
    arch: str  # x86 or x64
    is_dll: bool
    compile_time: str
    sections: list[PESectionDetail]
    exports: list[PEExport]
    imports: list[PEImportDll]
    version_info: PEVersionInfo
    interesting_strings: list[str]
    is_packed: bool
    packing_indicators: list[str]
    summary: str  # Human-readable verdict


def analyze_pe(binary_path: str) -> PEAnalysis:
    """Fast PE analysis from raw file. No Ghidra needed."""
    import pefile

    path = Path(binary_path)
    pe = pefile.PE(binary_path)

    # Architecture
    arch = "x64" if pe.FILE_HEADER.Machine == 0x8664 else "x86"
    is_dll = bool(pe.FILE_HEADER.Characteristics & 0x2000)

    # Compile time
    ts = pe.FILE_HEADER.TimeDateStamp
    try:
        compile_time = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (OSError, ValueError):
        compile_time = f"raw: {ts}"

    # Sections
    sections = []
    empty_code = []
    high_entropy = []
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        is_exec = bool(s.Characteristics & 0x20000000)
        is_empty = s.Misc_VirtualSize > 0 and s.SizeOfRawData == 0

        entropy = 0.0
        if s.SizeOfRawData > 0:
            data = s.get_data()
            counts = Counter(data)
            total = len(data)
            if total > 0:
                entropy = -sum(
                    (c / total) * math.log2(c / total)
                    for c in counts.values() if c > 0
                )

        sections.append(PESectionDetail(
            name=name,
            virtual_size=s.Misc_VirtualSize,
            raw_size=s.SizeOfRawData,
            entropy=entropy,
            is_executable=is_exec,
            is_empty=is_empty,
        ))

        if is_empty and (is_exec or name in (".text", ".code")):
            empty_code.append(name)
        if entropy > 7.0 and s.SizeOfRawData > 4096:
            high_entropy.append(name)

    # Packing detection
    packing_indicators = []
    is_packed = False
    if empty_code:
        is_packed = True
        packing_indicators.append(f"Empty executable sections: {', '.join(empty_code)}")
    if high_entropy:
        is_packed = True
        packing_indicators.append(f"High-entropy sections: {', '.join(high_entropy)}")

    ep_rva = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        if s.VirtualAddress <= ep_rva < s.VirtualAddress + s.Misc_VirtualSize:
            if name not in (".text", ".code", "CODE"):
                packing_indicators.append(f"Entry point in unusual section: {name}")
            break

    # Exports
    exports = []
    if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
            name = exp.name.decode() if exp.name else f"ordinal_{exp.ordinal}"
            exports.append(PEExport(name=name, ordinal=exp.ordinal, rva=exp.address))

    # Imports
    imports = []
    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll = entry.dll.decode()
            funcs = []
            for imp in entry.imports:
                if imp.name:
                    funcs.append(imp.name.decode())
                else:
                    funcs.append(f"ordinal_{imp.ordinal}")
            imports.append(PEImportDll(dll=dll, functions=funcs))

    # Version info
    version_info = PEVersionInfo()
    if hasattr(pe, "VS_VERSIONINFO") and hasattr(pe, "FileInfo"):
        for file_info in pe.FileInfo:
            for entry in file_info:
                if hasattr(entry, "StringTable"):
                    for st in entry.StringTable:
                        for key, value in st.entries.items():
                            k = key.decode() if isinstance(key, bytes) else key
                            v = value.decode() if isinstance(value, bytes) else value
                            if k == "CompanyName":
                                version_info.company = v
                            elif k == "FileDescription":
                                version_info.description = v
                            elif k == "FileVersion":
                                version_info.version = v
                            elif k == "ProductName":
                                version_info.product_name = v
                            elif k == "LegalCopyright":
                                version_info.copyright = v
                            elif k == "OriginalFilename":
                                version_info.original_filename = v

    pe.close()

    # Interesting strings (from raw file)
    interesting = _extract_interesting_strings(binary_path)

    # Build summary
    summary = _build_summary(
        path.name, arch, is_dll, compile_time, exports, imports,
        version_info, is_packed, packing_indicators, interesting,
    )

    return PEAnalysis(
        filename=path.name,
        file_size=path.stat().st_size,
        arch=arch,
        is_dll=is_dll,
        compile_time=compile_time,
        sections=sections,
        exports=exports,
        imports=imports,
        version_info=version_info,
        interesting_strings=interesting,
        is_packed=is_packed,
        packing_indicators=packing_indicators,
        summary=summary,
    )


def _extract_interesting_strings(binary_path: str) -> list[str]:
    """Extract notable strings from binary (URLs, paths, API names, etc.)."""
    import re
    data = Path(binary_path).read_bytes()

    # ASCII strings >= 6 chars
    ascii_strings = set(re.findall(rb"[\x20-\x7e]{6,}", data))
    # UTF-16LE strings >= 6 chars
    utf16_strings = set()
    for m in re.finditer(rb"(?:[\x20-\x7e]\x00){6,}", data):
        try:
            utf16_strings.add(m.group().decode("utf-16-le").encode("ascii"))
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

    all_strings = ascii_strings | utf16_strings

    interesting = set()
    patterns = [
        # Network
        rb"https?://",
        rb"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        # Files/paths
        rb"\.dll\b", rb"\.exe\b", rb"\.ini\b", rb"\.cfg\b", rb"\.log\b", rb"\.txt\b",
        rb"\\\\", rb"[A-Z]:\\",
        # Steam
        rb"(?i)steam", rb"(?i)valve", rb"(?i)goldberg",
        # Crypto/security
        rb"(?i)crypt", rb"(?i)cipher", rb"(?i)hash", rb"(?i)aes", rb"(?i)rsa",
        # System
        rb"(?i)registry", rb"(?i)mutex", rb"(?i)pipe",
        # Identity
        rb"(?i)copyright", rb"(?i)version", rb"(?i)author",
        # Suspicious
        rb"(?i)inject", rb"(?i)hook", rb"(?i)detour", rb"(?i)patch",
        rb"(?i)debug", rb"(?i)anti",
    ]

    for s in all_strings:
        for pattern in patterns:
            if re.search(pattern, s):
                try:
                    decoded = s.decode("ascii", errors="replace").strip()
                    if len(decoded) > 4 and not all(c in "!@#$%^&*(){}[]|\\<>?" for c in decoded):
                        interesting.add(decoded)
                except Exception:
                    pass
                break

    return sorted(interesting)[:200]


def _build_summary(
    filename: str, arch: str, is_dll: bool, compile_time: str,
    exports: list[PEExport], imports: list[PEImportDll],
    version_info: PEVersionInfo, is_packed: bool,
    packing_indicators: list[str], interesting_strings: list[str],
) -> str:
    """Build a human-readable summary of the analysis."""
    lines = []
    kind = "DLL" if is_dll else "EXE"
    lines.append(f"{filename}: {arch} Windows {kind}, compiled {compile_time}")

    if version_info.description:
        lines.append(f"Description: {version_info.description}")
    if version_info.company:
        lines.append(f"Company: {version_info.company}")
    if version_info.copyright:
        lines.append(f"Copyright: {version_info.copyright}")
    if version_info.version:
        lines.append(f"Version: {version_info.version}")

    if is_packed:
        lines.append(f"PACKED: {'; '.join(packing_indicators)}")

    if exports:
        # Categorize exports
        steam_exports = [e for e in exports if "Steam" in e.name or "Breakpad" in e.name]
        other_exports = [e for e in exports if e not in steam_exports]
        if steam_exports:
            lines.append(f"Steam API exports: {len(steam_exports)} (Steam API emulator/replacement)")
        if other_exports:
            lines.append(f"Other exports: {', '.join(e.name for e in other_exports[:10])}")

    if imports:
        dll_names = [i.dll for i in imports]
        lines.append(f"Imports from: {', '.join(dll_names)}")

    # Look for notable string patterns
    steam_strings = [s for s in interesting_strings if "steam" in s.lower() or "Steam" in s]
    url_strings = [s for s in interesting_strings if s.startswith("http")]
    if steam_strings:
        lines.append(f"Steam-related strings: {len(steam_strings)}")
    if url_strings:
        lines.append(f"URLs found: {', '.join(url_strings[:5])}")

    return "\n".join(lines)


def format_report(analysis: PEAnalysis) -> str:
    """Format a full analysis report as text."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  PE Analysis: {analysis.filename}")
    lines.append(f"{'=' * 70}")
    lines.append("")

    lines.append(f"  File size:    {analysis.file_size:,} bytes ({analysis.file_size / 1024 / 1024:.1f} MB)")
    lines.append(f"  Architecture: {analysis.arch}")
    lines.append(f"  Type:         {'DLL' if analysis.is_dll else 'EXE'}")
    lines.append(f"  Compiled:     {analysis.compile_time}")

    vi = analysis.version_info
    if vi.description or vi.company:
        lines.append("")
        lines.append("  Version Info:")
        if vi.description:
            lines.append(f"    Description: {vi.description}")
        if vi.company:
            lines.append(f"    Company:     {vi.company}")
        if vi.copyright:
            lines.append(f"    Copyright:   {vi.copyright}")
        if vi.version:
            lines.append(f"    Version:     {vi.version}")
        if vi.product_name:
            lines.append(f"    Product:     {vi.product_name}")

    if analysis.is_packed:
        lines.append("")
        lines.append("  PACKING DETECTED:")
        for ind in analysis.packing_indicators:
            lines.append(f"    - {ind}")

    lines.append("")
    lines.append("  Sections:")
    for s in analysis.sections:
        tag = ""
        if s.is_empty:
            tag = " [EMPTY]"
        elif s.entropy > 7.0:
            tag = " [ENCRYPTED]"
        lines.append(
            f"    {s.name:10s} VSize=0x{s.virtual_size:08x} "
            f"Raw=0x{s.raw_size:08x} Entropy={s.entropy:.2f}{tag}"
        )

    if analysis.exports:
        lines.append("")
        lines.append(f"  Exports ({len(analysis.exports)}):")
        for exp in analysis.exports:
            lines.append(f"    {exp.name} (ordinal {exp.ordinal})")

    if analysis.imports:
        lines.append("")
        lines.append(f"  Imports:")
        for imp in analysis.imports:
            lines.append(f"    {imp.dll}: {', '.join(imp.functions[:8])}")
            if len(imp.functions) > 8:
                lines.append(f"      ... +{len(imp.functions) - 8} more")

    if analysis.interesting_strings:
        lines.append("")
        lines.append(f"  Notable Strings ({len(analysis.interesting_strings)}):")
        for s in analysis.interesting_strings[:50]:
            lines.append(f"    {s}")
        if len(analysis.interesting_strings) > 50:
            lines.append(f"    ... +{len(analysis.interesting_strings) - 50} more")

    lines.append("")
    lines.append(f"{'=' * 70}")
    lines.append("  Summary")
    lines.append(f"{'=' * 70}")
    lines.append("")
    for line in analysis.summary.split("\n"):
        lines.append(f"  {line}")
    lines.append("")

    return "\n".join(lines)
