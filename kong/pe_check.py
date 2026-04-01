"""Fast PE packing detection from raw file — no Ghidra needed."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PEPackingResult:
    is_packed: bool = False
    confidence: float = 0.0
    indicators: list[str] = field(default_factory=list)


def check_pe_packing(binary_path: str) -> PEPackingResult:
    """Quickly check if a PE file is packed by examining section headers.

    This runs in milliseconds — no Ghidra, no JVM, just reads the PE headers.
    """
    try:
        import pefile
    except ImportError:
        return PEPackingResult()  # Can't check without pefile

    path = Path(binary_path)
    if not path.exists() or path.suffix.lower() not in (".dll", ".exe", ".sys"):
        return PEPackingResult()

    try:
        pe = pefile.PE(binary_path, fast_load=True)
    except Exception:
        return PEPackingResult()

    result = PEPackingResult()
    empty_code_sections: list[str] = []
    high_entropy_sections: list[str] = []

    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        is_exec = bool(s.Characteristics & 0x20000000)  # IMAGE_SCN_MEM_EXECUTE

        # Section has virtual size but no raw data = will be populated at runtime
        if s.Misc_VirtualSize > 0 and s.SizeOfRawData == 0:
            if is_exec or name in (".text", ".code"):
                empty_code_sections.append(name)
                result.indicators.append(
                    f"Section '{name}' is executable with virtual size "
                    f"0x{s.Misc_VirtualSize:x} but empty on disk"
                )

        # High entropy = encrypted or compressed
        if s.SizeOfRawData > 4096:
            data = s.get_data()
            counts = Counter(data)
            total = len(data)
            entropy = -sum(
                (c / total) * math.log2(c / total)
                for c in counts.values() if c > 0
            )
            if entropy > 7.0:
                high_entropy_sections.append(name)
                result.indicators.append(
                    f"Section '{name}' has entropy {entropy:.2f}/8.0 "
                    f"({s.SizeOfRawData} bytes) — likely encrypted/compressed"
                )

    # Check entry point location
    ep_rva = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        if s.VirtualAddress <= ep_rva < s.VirtualAddress + s.Misc_VirtualSize:
            if name not in (".text", ".code", "CODE"):
                result.indicators.append(
                    f"Entry point is in section '{name}' (not .text)"
                )
            break

    pe.close()

    # Only flag as packed if actual code sections are empty.
    # High-entropy non-code sections (packer stubs, resources) alone aren't a problem
    # as long as .text has real code.
    text_has_code = any(
        name in (".text", ".code") and s.SizeOfRawData > 0
        for s in pe.sections
        for name in [s.Name.rstrip(b"\x00").decode("ascii", errors="replace")]
    )
    if empty_code_sections:
        result.is_packed = True
        result.confidence = 0.95
    elif high_entropy_sections and not text_has_code:
        result.is_packed = True
        result.confidence = 0.7

    return result
