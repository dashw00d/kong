"""Automatic unpacking of packed/encrypted PE binaries via Speakeasy emulation.

Uses Mandiant's Speakeasy framework to emulate the binary, let the unpacker
stub execute, and dump the unpacked code sections to a new PE file that
Ghidra can analyze meaningfully.
"""

from __future__ import annotations

import logging
import struct
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class UnpackError(Exception):
    """Raised when unpacking fails."""


def unpack_pe(binary_path: str, output_path: str | None = None, timeout: int = 60) -> str:
    """Attempt to unpack a PE binary using Speakeasy emulation.

    Args:
        binary_path: Path to the packed PE file.
        output_path: Where to write the unpacked PE. If None, writes next to the original.
        timeout: Maximum emulation time in seconds.

    Returns:
        Path to the unpacked PE file.

    Raises:
        UnpackError: If unpacking fails.
    """
    try:
        import speakeasy
    except ImportError:
        raise UnpackError(
            "speakeasy-emulator is not installed. "
            "Install it with: pip install speakeasy-emulator"
        )

    import pefile

    binary_path = str(Path(binary_path).resolve())
    logger.info("Attempting to unpack: %s", binary_path)

    # Validate input is a PE with signs of packing
    try:
        pe = pefile.PE(binary_path)
    except Exception as e:
        raise UnpackError(f"Not a valid PE file: {e}")

    # Find sections that are empty on disk but have virtual size
    empty_sections = []
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        if s.Misc_VirtualSize > 0 and s.SizeOfRawData == 0:
            empty_sections.append(name)

    if not empty_sections:
        raise UnpackError("Binary doesn't appear to be packed (no empty sections)")

    logger.info("Empty sections that should be populated: %s", empty_sections)

    # Run through Speakeasy
    se = speakeasy.Speakeasy()

    # Track memory writes to detect unpacked regions
    written_regions: dict[int, bytearray] = {}
    original_sections: dict[str, tuple[int, int]] = {}  # name -> (rva, vsize)

    image_base = pe.OPTIONAL_HEADER.ImageBase
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        original_sections[name] = (s.VirtualAddress, s.Misc_VirtualSize)

    pe.close()

    try:
        module = se.load_module(binary_path)
    except Exception as e:
        raise UnpackError(f"Speakeasy failed to load module: {e}")

    try:
        se.run_module(module, all_entrypoints=True)
    except Exception as e:
        # Emulation errors are expected — the unpacker may hit unsupported APIs.
        # That's fine as long as the sections got populated before the crash.
        logger.info("Emulation ended: %s (this is often expected)", e)

    # Read back the populated sections from Speakeasy's memory
    emu = se.emu
    populated = {}
    for name, (rva, vsize) in original_sections.items():
        if name not in empty_sections:
            continue
        addr = image_base + rva
        try:
            data = emu.mem_read(addr, vsize)
            # Check if the section actually got populated (not all zeros)
            if any(b != 0 for b in data[:4096]):
                populated[name] = bytes(data)
                nonzero = sum(1 for b in data if b != 0)
                logger.info(
                    "Section '%s' populated: %d bytes (%d%% non-zero)",
                    name, vsize, (nonzero * 100) // vsize,
                )
            else:
                logger.warning("Section '%s' still empty after emulation", name)
        except Exception as e:
            logger.warning("Could not read section '%s' from emulator: %s", name, e)

    if not populated:
        raise UnpackError(
            "Emulation completed but no sections were populated. "
            "The packer may use anti-emulation techniques. "
            "Try dumping manually with pe-sieve or Scylla instead."
        )

    # Build the unpacked PE by patching the original
    unpacked_pe = _rebuild_pe(binary_path, populated, original_sections)

    if output_path is None:
        p = Path(binary_path)
        output_path = str(p.parent / f"{p.stem}_unpacked{p.suffix}")

    Path(output_path).write_bytes(unpacked_pe)
    logger.info("Unpacked binary written to: %s", output_path)

    # Validate the result
    try:
        check = pefile.PE(output_path)
        text_section = None
        for s in check.sections:
            name = s.Name.rstrip(b"\x00").decode("ascii", errors="replace")
            if name == ".text":
                text_section = s
                break
        if text_section and text_section.SizeOfRawData > 0:
            logger.info(
                "Validation OK: .text section now has %d bytes of raw data",
                text_section.SizeOfRawData,
            )
        check.close()
    except Exception as e:
        logger.warning("Unpacked PE validation warning: %s", e)

    return output_path


def _rebuild_pe(
    original_path: str,
    populated: dict[str, bytes],
    section_info: dict[str, tuple[int, int]],
) -> bytes:
    """Rebuild a PE file with populated sections replacing empty ones.

    Takes the original PE, expands the empty sections to hold the dumped data,
    and adjusts file offsets/sizes accordingly.
    """
    import pefile

    pe = pefile.PE(original_path)

    for section in pe.sections:
        name = section.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        if name in populated:
            data = populated[name]
            # Align to file alignment
            file_alignment = pe.OPTIONAL_HEADER.FileAlignment
            aligned_size = (len(data) + file_alignment - 1) & ~(file_alignment - 1)

            section.SizeOfRawData = aligned_size
            section.Misc_VirtualSize = len(data)

    # Now we need to recalculate all PointerToRawData offsets
    # Sort sections by VirtualAddress to maintain order
    sorted_sections = sorted(pe.sections, key=lambda s: s.VirtualAddress)

    # Start raw data after headers
    current_offset = pe.OPTIONAL_HEADER.SizeOfHeaders
    file_alignment = pe.OPTIONAL_HEADER.FileAlignment
    current_offset = (current_offset + file_alignment - 1) & ~(file_alignment - 1)

    for section in sorted_sections:
        section.PointerToRawData = current_offset
        current_offset += section.SizeOfRawData

    # Build the output
    pe_data = bytearray(pe.write())

    # Now overlay the populated section data at the correct offsets
    for section in sorted_sections:
        name = section.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        if name in populated:
            data = populated[name]
            offset = section.PointerToRawData
            # Extend if needed
            needed = offset + section.SizeOfRawData
            if needed > len(pe_data):
                pe_data.extend(b"\x00" * (needed - len(pe_data)))
            pe_data[offset:offset + len(data)] = data

    pe.close()
    return bytes(pe_data)
