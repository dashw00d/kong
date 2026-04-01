"""Ghidra client — in-process via PyGhidra/JPype.

Opens a binary directly in the current process using PyGhidra, then calls
Ghidra Java APIs through JPype.  No subprocess, no RPC, no port management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyghidra

from kong.ghidra.types import (
    BasicBlock,
    BinaryInfo,
    BlockEdge,
    BlockEdgeType,
    ControlFlowGraph,
    FunctionClassification,
    FunctionInfo,
    PackingInfo,
    ParameterInfo,
    PcodeOp,
    SectionInfo,
    StringEntry,
    StructDefinition,
    StructField,
    VariableInfo,
    XRef,
)
from kong.ghidra.environment import find_ghidra_install


logger = logging.getLogger(__name__)


class GhidraClientError(Exception):
    """Raised when a Ghidra operation fails."""


def _classify(size: int, is_thunk: bool) -> FunctionClassification:
    if is_thunk:
        return FunctionClassification.THUNK
    if size <= 16:
        return FunctionClassification.TRIVIAL
    if size <= 64:
        return FunctionClassification.SMALL
    if size <= 256:
        return FunctionClassification.MEDIUM
    return FunctionClassification.LARGE


class GhidraClient:
    """In-process Ghidra client via PyGhidra.

    Usage::

        with GhidraClient("/path/to/binary") as client:
            funcs = client.list_functions()
            decomp = client.get_decompilation(funcs[0].address)
    """

    def __init__(
        self,
        binary_path: str,
        install_dir: str | None = None,
        jvm_heap: str = "4g",
    ) -> None:
        self.binary_path = str(Path(binary_path).resolve())
        if install_dir is None:
            install_dir = find_ghidra_install()
        self.install_dir = install_dir
        self.jvm_heap = jvm_heap
        self._program: Any = None
        self._flat_api: Any = None
        self._project: Any = None
        self._decomp_interface: Any = None

    @property
    def program(self) -> Any:
        if self._program is None:
            raise GhidraClientError("Not open. Call open() first.")
        return self._program

    @property
    def flat_api(self) -> Any:
        if self._flat_api is None:
            raise GhidraClientError("Not open. Call open() first.")
        return self._flat_api

    @property
    def decomp_interface(self) -> Any:
        """Lazily create and cache a shared DecompInterface."""
        if self._decomp_interface is None:
            from ghidra.app.decompiler import DecompInterface, DecompileOptions
            opts = DecompileOptions()
            opts.setMaxInstructions(100000)
            opts.setMaxPayloadMBytes(100)
            di = DecompInterface()
            di.setSimplificationStyle("decompile")
            di.setOptions(opts)
            di.openProgram(self.program)
            self._decomp_interface = di
        return self._decomp_interface

    def open(self) -> GhidraClient:
        """Start PyGhidra JVM and open the binary for analysis."""
        if not Path(self.binary_path).exists():
            raise GhidraClientError(f"Binary not found: {self.binary_path}")

        logger.info("Starting PyGhidra JVM ...")
        if not pyghidra.started():
            from pyghidra.launcher import HeadlessPyGhidraLauncher
            launcher = HeadlessPyGhidraLauncher(install_dir=self.install_dir)
            launcher.add_vmargs(f"-Xmx{self.jvm_heap}")
            launcher.start()
        else:
            pyghidra.start(install_dir=self.install_dir)

        # Manage the Ghidra project lifecycle directly (not via pyghidra's
        # context manager) so we can call project.save(program) at any time.
        from pyghidra.core import _setup_project, _analyze_program
        from ghidra.program.flatapi import FlatProgramAPI
        from ghidra.app.script import GhidraScriptUtil

        logger.info("Opening program: %s", self.binary_path)
        binary_name = Path(self.binary_path).stem
        project_location = Path(self.binary_path).parent / f"{binary_name}_kong"
        self._clean_stale_locks(project_location)

        self._project, self._program = _setup_project(
            self.binary_path,
            project_location=str(project_location),
            project_name="kong",
        )
        GhidraScriptUtil.acquireBundleHostReference()
        self._flat_api = FlatProgramAPI(self._program)

        # Check if analysis is needed
        func_count = self._program.getFunctionManager().getFunctionCount()
        if func_count < 100:
            if func_count > 0:
                logger.warning(
                    "Project has only %d functions — incomplete. Re-analyzing...",
                    func_count,
                )
            else:
                logger.info("New project — running Ghidra auto-analysis (this is slow)")
            _analyze_program(self._flat_api, self._program)
            self.save_project()
            func_count = self._program.getFunctionManager().getFunctionCount()

        logger.info(
            "Program loaded: %s (%d functions)",
            self._program.getName(), func_count,
        )
        return self

    def save_project(self) -> None:
        """Flush the Ghidra project to disk so progress survives kills."""
        if self._project is None or self._program is None:
            return
        try:
            self._project.save(self._program)
            logger.info("Ghidra project saved to disk")
        except Exception:
            logger.debug("Failed to save Ghidra project", exc_info=True)

    def close(self) -> None:
        """Close the program and release resources."""
        if self._decomp_interface is not None:
            try:
                self._decomp_interface.dispose()
            except Exception:
                logger.debug("Error disposing DecompInterface", exc_info=True)
            self._decomp_interface = None
        if self._project is not None:
            try:
                self._project.save(self._program)
            except Exception:
                logger.debug("Error saving Ghidra project on close", exc_info=True)
            try:
                from ghidra.app.script import GhidraScriptUtil
                GhidraScriptUtil.releaseBundleHostReference()
            except Exception:
                pass
            try:
                self._project.close()
            except Exception:
                logger.debug("Error closing Ghidra project", exc_info=True)
            self._project = None
        self._program = None
        self._flat_api = None

    @staticmethod
    def _clean_stale_locks(project_location: Path) -> None:
        """Remove stale .lock files from a Ghidra project directory.

        Ghidra creates .lock files when a project is open. If the process
        crashes, these persist and block the next run from opening the project.
        """
        if not project_location.exists():
            return
        for lock_file in project_location.rglob("*.lock"):
            logger.info("Removing stale lock file: %s", lock_file)
            try:
                lock_file.unlink()
            except OSError:
                logger.warning("Could not remove lock file: %s", lock_file)

    def __enter__(self) -> GhidraClient:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_binary_info(self) -> BinaryInfo:
        """Get metadata about the loaded binary."""
        prog = self.program
        lang = prog.getLanguage()
        sections = self.get_sections()
        packing = self.detect_packing(sections)
        return BinaryInfo(
            name=str(prog.getName()),
            path=str(prog.getExecutablePath()),
            arch=str(lang.getProcessor().toString()),
            endianness="big" if lang.isBigEndian() else "little",
            word_size=int(prog.getDefaultPointerSize()),
            format=str(prog.getExecutableFormat()),
            compiler=str(prog.getCompilerSpec().getCompilerSpecID()),
            min_address=int(prog.getMinAddress().getOffset()),
            max_address=int(prog.getMaxAddress().getOffset()),
            sections=sections,
            packing=packing,
        )

    def get_sections(self) -> list[SectionInfo]:
        """Get PE sections / memory blocks from the binary."""
        import math
        from collections import Counter

        memory = self.program.getMemory()
        sections: list[SectionInfo] = []
        for block in memory.getBlocks():
            name = str(block.getName())
            start = int(block.getStart().getOffset())
            size = int(block.getSize())
            is_exec = bool(block.isExecute())
            is_init = bool(block.isInitialized())

            entropy = 0.0
            if is_init and size > 0:
                try:
                    raw = bytearray(size)
                    block.getBytes(block.getStart(), raw)
                    counts = Counter(raw)
                    total = len(raw)
                    entropy = -sum(
                        (c / total) * math.log2(c / total)
                        for c in counts.values() if c > 0
                    )
                except Exception:
                    pass

            sections.append(SectionInfo(
                name=name,
                virtual_address=start,
                virtual_size=size,
                raw_size=size if is_init else 0,
                is_executable=is_exec,
                is_initialized=is_init,
                entropy=entropy,
            ))
        return sections

    def detect_packing(self, sections: list[SectionInfo]) -> PackingInfo:
        """Analyze sections for signs of packing/encryption."""
        info = PackingInfo()

        # Check for empty code sections (VSize > 0 but uninitialized)
        code_section_names = {".text", ".code", "CODE"}
        for s in sections:
            if s.name in code_section_names or (s.is_executable and s.name.startswith(".")):
                if not s.is_initialized and s.virtual_size > 0:
                    info.empty_code_sections.append(s.name)
                    info.indicators.append(
                        f"Section '{s.name}' has virtual size 0x{s.virtual_size:x} "
                        f"but no raw data (uninitialized)"
                    )

        # Check for high-entropy sections (> 7.0 = likely encrypted/compressed)
        for s in sections:
            if s.entropy > 7.0 and s.raw_size > 1024:
                info.high_entropy_sections.append(s.name)
                info.indicators.append(
                    f"Section '{s.name}' has very high entropy ({s.entropy:.2f}/8.0) — "
                    f"likely encrypted or compressed"
                )

        # Check if entry point is in an unusual section
        ep = int(self.program.getImageBase().getOffset()) + \
            int(self.program.getMemory().getMinAddress().getOffset())
        # Actually, let's get the real entry point from the listing
        try:
            entry_func = None
            fm = self.program.getFunctionManager()
            for sym in self.program.getSymbolTable().getExternalEntryPointIterator():
                entry_func = fm.getFunctionAt(sym)
                if entry_func:
                    break
            if entry_func is None:
                # Try the conventional entry point
                entry_addr = self.program.getMinAddress()
                for func in fm.getFunctions(True):
                    if "entry" in str(func.getName()).lower():
                        entry_func = func
                        break
            if entry_func:
                ep_addr = int(entry_func.getEntryPoint().getOffset())
                for s in sections:
                    if s.virtual_address <= ep_addr < s.virtual_address + s.virtual_size:
                        info.entry_section = s.name
                        if s.name not in (".text", ".code", "CODE"):
                            info.indicators.append(
                                f"Entry point is in unusual section '{s.name}' "
                                f"(expected .text)"
                            )
                        break
        except Exception:
            pass

        # Determine if packed
        if info.empty_code_sections:
            info.is_packed = True
            info.confidence = 0.95
        elif len(info.high_entropy_sections) >= 2:
            info.is_packed = True
            info.confidence = 0.8
        elif info.high_entropy_sections and info.entry_section not in (".text", ".code", "CODE", ""):
            info.is_packed = True
            info.confidence = 0.7

        return info

    def list_functions(self, sections: list[SectionInfo] | None = None) -> list[FunctionInfo]:
        """List all functions in the binary.

        If sections are provided, functions in non-standard sections
        are classified as PACKED_SECTION so they can be skipped.
        """
        junk_ranges: list[tuple[int, int]] = []
        if sections:
            # Standard PE sections that contain real analyzable code/data
            standard_sections = {
                ".text", ".code", "CODE", ".rdata", ".data", ".bss",
                ".idata", ".edata", ".pdata", ".rsrc", ".reloc",
                ".tls", ".gfids", ".gehcont", ".00cfg",
            }
            for s in sections:
                if s.name not in standard_sections:
                    junk_ranges.append(
                        (s.virtual_address, s.virtual_address + s.virtual_size)
                    )
                    logger.info(
                        "Marking section '%s' (0x%x-0x%x) as non-standard — "
                        "functions will be skipped",
                        s.name, s.virtual_address,
                        s.virtual_address + s.virtual_size,
                    )

        def _in_junk(addr: int) -> bool:
            return any(start <= addr < end for start, end in junk_ranges)

        fm = self.program.getFunctionManager()
        functions: list[FunctionInfo] = []
        junk_count = 0
        for func in fm.getFunctions(True):
            addr = int(func.getEntryPoint().getOffset())
            size = int(func.getBody().getNumAddresses())
            is_thunk = bool(func.isThunk())
            if junk_ranges and _in_junk(addr):
                classification = FunctionClassification.PACKED_SECTION
                junk_count += 1
            else:
                classification = _classify(size, is_thunk)
            functions.append(
                FunctionInfo(
                    address=addr,
                    name=str(func.getName()),
                    size=size,
                    is_thunk=is_thunk,
                    classification=classification,
                )
            )

        if junk_count:
            logger.info(
                "Filtered %d/%d functions in non-standard sections",
                junk_count, len(functions),
            )
        return functions

    def get_function_info(self, addr: int) -> FunctionInfo:
        """Get detailed information about a specific function."""
        func = self._get_function_at(addr)
        size = int(func.getBody().getNumAddresses())
        is_thunk = bool(func.isThunk())

        params = [
            ParameterInfo(
                name=str(p.getName()),
                data_type=str(p.getDataType().getDisplayName()),
                ordinal=int(p.getOrdinal()),
                size=int(p.getLength()),
            )
            for p in func.getParameters()
        ]
        local_vars = [
            VariableInfo(
                name=str(v.getName()),
                data_type=str(v.getDataType().getDisplayName()),
                size=int(v.getLength()),
                stack_offset=int(v.getStackOffset()) if v.isStackVariable() else None,
            )
            for v in func.getLocalVariables()
        ]

        return FunctionInfo(
            address=int(func.getEntryPoint().getOffset()),
            name=str(func.getName()),
            size=size,
            is_thunk=is_thunk,
            params=params,
            return_type=str(func.getReturnType().getDisplayName()),
            local_vars=local_vars,
            calling_convention=str(func.getCallingConventionName()),
            classification=_classify(size, is_thunk),
        )

    def get_decompilation(self, addr: int, timeout: int = 120) -> str:
        """Get the decompiled C source for a function."""
        from ghidra.util.task import ConsoleTaskMonitor

        func = self._get_function_at(addr)
        result = self.decomp_interface.decompileFunction(
            func, timeout, ConsoleTaskMonitor()
        )
        if not result.decompileCompleted():
            err = result.getErrorMessage() or "unknown error"
            raise GhidraClientError(
                f"Decompilation failed for function at 0x{addr:08x}: {err}"
            )
        decomp_func = result.getDecompiledFunction()
        if decomp_func is None:
            err = result.getErrorMessage() or "no decompiled output"
            raise GhidraClientError(
                f"Decompilation failed for function at 0x{addr:08x}: {err}"
            )
        return str(decomp_func.getC())

    def get_xrefs_to(self, addr: int) -> list[XRef]:
        """Get all cross-references TO a given address."""
        target = self._to_addr(addr)
        refs = self.flat_api.getReferencesTo(target)
        return [
            XRef(
                from_addr=int(ref.getFromAddress().getOffset()),
                to_addr=int(ref.getToAddress().getOffset()),
                ref_type=str(ref.getReferenceType().getName()),
            )
            for ref in refs
        ]

    def get_xrefs_from(self, addr: int) -> list[XRef]:
        """Get all cross-references FROM an address."""
        source = self._to_addr(addr)
        refs = self.program.getReferenceManager().getReferencesFrom(source)
        return [
            XRef(
                from_addr=int(ref.getFromAddress().getOffset()),
                to_addr=int(ref.getToAddress().getOffset()),
                ref_type=str(ref.getReferenceType().getName()),
            )
            for ref in refs
        ]

    def get_callers(self, addr: int) -> list[int]:
        """Get addresses of functions that call the function at addr."""
        target = self._to_addr(addr)
        refs = self.flat_api.getReferencesTo(target)
        return list({
            int(ref.getFromAddress().getOffset())
            for ref in refs
            if ref.getReferenceType().isCall()
        })

    def get_callees(self, addr: int) -> list[int]:
        """Get addresses of functions called by the function at addr."""
        source = self._to_addr(addr)
        refs = self.program.getReferenceManager().getReferencesFrom(source)
        return list({
            int(ref.getToAddress().getOffset())
            for ref in refs
            if ref.getReferenceType().isCall()
        })

    def get_strings(self) -> list[StringEntry]:
        """Get all defined strings in the binary."""
        prog = self.program
        listing = prog.getListing()
        entries: list[StringEntry] = []
        for data in listing.getDefinedData(True):
            dt_name = str(data.getDataType().getName()).lower()
            if "string" not in dt_name:
                continue
            addr = data.getAddress()
            addr_offset = int(addr.getOffset())
            value = data.getValue()
            xrefs = [
                int(ref.getFromAddress().getOffset())
                for ref in self.flat_api.getReferencesTo(addr)
            ]
            entries.append(
                StringEntry(
                    address=addr_offset,
                    value=str(value) if value is not None else "",
                    length=int(data.getLength()),
                    xref_addrs=xrefs,
                )
            )
        return entries

    def rename_function(self, addr: int, new_name: str) -> None:
        """Rename a function at the given address."""
        func = self._get_function_at(addr)
        from ghidra.program.model.symbol import SourceType
        tx = self.program.startTransaction("rename_function")
        try:
            func.setName(new_name, SourceType.USER_DEFINED)
        finally:
            self.program.endTransaction(tx, True)
        logger.info("Renamed function at 0x%08x to '%s'", addr, new_name)

    def set_function_signature(self, addr: int, signature_str: str) -> None:
        """Set a function's full signature from a C-style string.

        Preserves the existing calling convention so Ghidra doesn't warn about
        unknown convention with locked parameter storage.
        """
        from ghidra.app.cmd.function import ApplyFunctionSignatureCmd
        from ghidra.app.util.parser import FunctionSignatureParser
        from ghidra.program.model.symbol import SourceType

        func = self._get_function(addr)
        calling_convention = func.getCallingConventionName()

        dtm = self.program.getDataTypeManager()
        parser = FunctionSignatureParser(dtm, None)
        func_def = parser.parse(None, signature_str)

        if calling_convention:
            func_def.setCallingConvention(calling_convention)

        cmd = ApplyFunctionSignatureCmd(
            self._to_addr(addr),
            func_def,
            SourceType.USER_DEFINED,
            True,   # preserveCallingConvention
            False,  # forceSetName
        )
        tx = self.program.startTransaction("set_function_signature")
        try:
            cmd.applyTo(self.program)
        finally:
            self.program.endTransaction(tx, True)
        logger.info("Set signature at 0x%08x to '%s'", addr, signature_str)

    def add_comment(
        self,
        addr: int,
        comment: str,
        comment_type: str = "plate",
    ) -> None:
        """Add a comment to an address.

        Args:
            addr: Address to comment.
            comment: Comment text.
            comment_type: One of "plate", "pre", "post", "eol", "repeatable".
        """
        from ghidra.program.model.listing import CodeUnit

        type_map = {
            "plate": CodeUnit.PLATE_COMMENT,
            "pre": CodeUnit.PRE_COMMENT,
            "post": CodeUnit.POST_COMMENT,
            "eol": CodeUnit.EOL_COMMENT,
            "repeatable": CodeUnit.REPEATABLE_COMMENT,
        }
        if comment_type not in type_map:
            raise ValueError(
                f"Invalid comment_type: {comment_type}. "
                f"Use one of {list(type_map.keys())}"
            )

        code_unit = self.program.getListing().getCodeUnitAt(self._to_addr(addr))
        tx = self.program.startTransaction("add_comment")
        try:
            code_unit.setComment(type_map[comment_type], comment)
        finally:
            self.program.endTransaction(tx, True)
        logger.info("Added %s comment at 0x%08x", comment_type, addr)

    def create_struct(self, definition: StructDefinition) -> None:
        """Create a struct data type in Ghidra's DataTypeManager.

        Fields are placed at explicit offsets within the struct. Gaps between
        fields are left as undefined bytes (Ghidra fills them automatically).
        """
        from ghidra.program.model.data import (
            CategoryPath,
            StructureDataType,
        )

        dtm = self.program.getDataTypeManager()
        category = CategoryPath("/kong")

        struct_dt = StructureDataType(category, definition.name, definition.size)
        for fld in definition.fields:
            ghidra_type = self._resolve_data_type(fld.data_type, fld.size)
            struct_dt.replaceAtOffset(fld.offset, ghidra_type, fld.size, fld.name, None)

        tx = self.program.startTransaction("create_struct")
        try:
            dtm.addDataType(struct_dt, None)
        finally:
            self.program.endTransaction(tx, True)
        logger.info("Created struct '%s' (%d bytes, %d fields)", definition.name, definition.size, definition.field_count)

    def apply_type_to_param(
        self,
        func_addr: int,
        param_ordinal: int,
        type_name: str,
        as_pointer: bool = True,
    ) -> None:
        """Apply a data type (optionally as a pointer) to a function parameter."""
        from ghidra.program.model.data import PointerDataType
        from ghidra.program.model.symbol import SourceType

        func = self._get_function_at(func_addr)
        params = func.getParameters()
        if param_ordinal >= len(params):
            raise GhidraClientError(
                f"Parameter ordinal {param_ordinal} out of range for "
                f"function at 0x{func_addr:08x} ({len(params)} params)"
            )

        resolved = self._lookup_type(type_name)
        if resolved is None:
            raise GhidraClientError(f"Type '{type_name}' not found in DataTypeManager")

        target_type = PointerDataType(resolved) if as_pointer else resolved

        tx = self.program.startTransaction("apply_type_to_param")
        try:
            param = params[param_ordinal]
            param.setDataType(target_type, SourceType.USER_DEFINED)
        finally:
            self.program.endTransaction(tx, True)
        logger.info(
            "Applied type '%s%s' to param %d of function at 0x%08x",
            type_name, " *" if as_pointer else "", param_ordinal, func_addr,
        )

    def list_custom_types(self) -> list[StructDefinition]:
        """List all struct types in the /kong category."""
        from ghidra.program.model.data import CategoryPath, Structure

        dtm = self.program.getDataTypeManager()
        category = dtm.getCategory(CategoryPath("/kong"))
        if category is None:
            return []

        results: list[StructDefinition] = []
        for dt in category.getDataTypes():
            if not isinstance(dt, Structure):
                continue
            fields = []
            for i in range(dt.getNumDefinedComponents()):
                comp = dt.getComponent(i)
                fields.append(StructField(
                    name=str(comp.getFieldName() or f"field_{i}"),
                    data_type=str(comp.getDataType().getDisplayName()),
                    offset=int(comp.getOffset()),
                    size=int(comp.getLength()),
                ))
            results.append(StructDefinition(
                name=str(dt.getName()),
                size=int(dt.getLength()),
                fields=fields,
            ))
        return results

    def get_type(self, name: str) -> StructDefinition | None:
        """Look up a struct type by name from the /kong category."""
        from ghidra.program.model.data import Structure

        resolved = self._lookup_type(name)
        if resolved is None or not isinstance(resolved, Structure):
            return None

        fields = []
        for i in range(resolved.getNumDefinedComponents()):
            comp = resolved.getComponent(i)
            fields.append(StructField(
                name=str(comp.getFieldName() or f"field_{i}"),
                data_type=str(comp.getDataType().getDisplayName()),
                offset=int(comp.getOffset()),
                size=int(comp.getLength()),
            ))
        return StructDefinition(
            name=str(resolved.getName()),
            size=int(resolved.getLength()),
            fields=fields,
        )

    def _lookup_type(self, name: str) -> Any:
        """Search for a data type by name, checking /kong category first."""
        from ghidra.program.model.data import CategoryPath
        from java.util import ArrayList

        dtm = self.program.getDataTypeManager()
        category = dtm.getCategory(CategoryPath("/kong"))
        if category is not None:
            for dt in category.getDataTypes():
                if str(dt.getName()) == name:
                    return dt

        results = ArrayList()
        dtm.findDataTypes(name, results)
        return results.get(0) if results.size() > 0 else None

    def _resolve_data_type(self, type_name: str, size: int) -> Any:
        """Resolve a C type name to a Ghidra DataType, falling back to sized defaults."""
        from ghidra.program.model.data import (
            ByteDataType,
            CharDataType,
            IntegerDataType,
            LongDataType,
            LongLongDataType,
            PointerDataType,
            ShortDataType,
            UnsignedIntegerDataType,
            UnsignedLongDataType,
            UnsignedLongLongDataType,
            UnsignedShortDataType,
            Undefined1DataType,
            Undefined2DataType,
            Undefined4DataType,
            Undefined8DataType,
        )

        _BUILTIN: dict[str, Any] = {
            "byte": ByteDataType.dataType,
            "char": CharDataType.dataType,
            "short": ShortDataType.dataType,
            "int": IntegerDataType.dataType,
            "long": LongDataType.dataType,
            "long long": LongLongDataType.dataType,
            "uint8_t": ByteDataType.dataType,
            "uint16_t": UnsignedShortDataType.dataType,
            "uint32_t": UnsignedIntegerDataType.dataType,
            "uint64_t": UnsignedLongLongDataType.dataType,
            "int8_t": CharDataType.dataType,
            "int16_t": ShortDataType.dataType,
            "int32_t": IntegerDataType.dataType,
            "int64_t": LongLongDataType.dataType,
            "unsigned int": UnsignedIntegerDataType.dataType,
            "unsigned long": UnsignedLongDataType.dataType,
            "unsigned short": UnsignedShortDataType.dataType,
        }

        if type_name.endswith("*"):
            base_name = type_name.rstrip("* ").strip()
            base_type = self._resolve_data_type(base_name, self.program.getDefaultPointerSize())
            return PointerDataType(base_type)

        low = type_name.lower().strip()
        if low in _BUILTIN:
            return _BUILTIN[low]

        found = self._lookup_type(type_name)
        if found is not None:
            return found

        _SIZED_DEFAULTS: dict[int, Any] = {
            1: Undefined1DataType.dataType,
            2: Undefined2DataType.dataType,
            4: Undefined4DataType.dataType,
            8: Undefined8DataType.dataType,
        }
        return _SIZED_DEFAULTS.get(size, Undefined4DataType.dataType)

    def get_basic_blocks(self, addr: int) -> list[BasicBlock]:
        """Get basic blocks for the function at addr."""
        from ghidra.program.model.block import SimpleBlockModel
        from ghidra.util.task import ConsoleTaskMonitor

        func = self._get_function_at(addr)
        body = func.getBody()
        model = SimpleBlockModel(self.program)
        monitor = ConsoleTaskMonitor()

        blocks: list[BasicBlock] = []
        block_iter = model.getCodeBlocksContaining(body, monitor)
        while block_iter.hasNext():
            block = block_iter.next()
            start = int(block.getFirstStartAddress().getOffset())
            end = int(block.getMaxAddress().getOffset())
            instructions = self._disassemble_range(start, end)
            blocks.append(BasicBlock(
                start_addr=start,
                end_addr=end,
                instructions=instructions,
            ))
        return blocks

    def get_control_flow_graph(self, addr: int) -> ControlFlowGraph:
        """Build a complete control flow graph for the function at addr."""
        from ghidra.program.model.block import SimpleBlockModel
        from ghidra.util.task import ConsoleTaskMonitor

        func = self._get_function_at(addr)
        body = func.getBody()
        model = SimpleBlockModel(self.program)
        monitor = ConsoleTaskMonitor()

        blocks: list[BasicBlock] = []
        edges: list[BlockEdge] = []
        block_iter = model.getCodeBlocksContaining(body, monitor)
        while block_iter.hasNext():
            block = block_iter.next()
            start = int(block.getFirstStartAddress().getOffset())
            end = int(block.getMaxAddress().getOffset())
            instructions = self._disassemble_range(start, end)
            blocks.append(BasicBlock(
                start_addr=start,
                end_addr=end,
                instructions=instructions,
            ))

            dest_iter = block.getDestinations(monitor)
            while dest_iter.hasNext():
                dest_ref = dest_iter.next()
                dest_block = dest_ref.getDestinationBlock()
                if dest_block is None:
                    continue
                dest_addr = int(dest_block.getFirstStartAddress().getOffset())
                if not body.contains(dest_block.getFirstStartAddress()):
                    continue
                flow_type = dest_ref.getFlowType()
                edge_type = self._classify_flow(flow_type)
                edges.append(BlockEdge(
                    from_addr=start,
                    to_addr=dest_addr,
                    edge_type=edge_type,
                ))

        return ControlFlowGraph(
            function_addr=int(func.getEntryPoint().getOffset()),
            blocks=blocks,
            edges=edges,
        )

    def get_pcode_ops(self, addr: int) -> list[PcodeOp]:
        """Get high-level pcode operations for the function at addr."""
        from ghidra.util.task import ConsoleTaskMonitor

        func = self._get_function_at(addr)
        result = self.decomp_interface.decompileFunction(
            func, 60, ConsoleTaskMonitor()
        )
        if not result.decompileCompleted():
            raise GhidraClientError(
                f"Decompilation failed for pcode at 0x{addr:08x}"
            )
        high_func = result.getHighFunction()
        if high_func is None:
            return []

        ops: list[PcodeOp] = []
        op_iter = high_func.getPcodeOps()
        while op_iter.hasNext():
            op = op_iter.next()
            mnemonic = str(op.getMnemonic())
            op_addr = int(op.getSeqnum().getTarget().getOffset())
            inputs = [
                str(op.getInput(i)) for i in range(op.getNumInputs())
            ]
            output = str(op.getOutput()) if op.getOutput() is not None else ""
            ops.append(PcodeOp(
                mnemonic=mnemonic,
                address=op_addr,
                inputs=inputs,
                output=output,
            ))
        return ops

    def get_disassembly(self, addr: int) -> str:
        """Get disassembly listing for a function as a formatted string."""
        func = self._get_function_at(addr)
        listing = self.program.getListing()
        body = func.getBody()
        lines: list[str] = []
        instr = listing.getInstructionAt(body.getMinAddress())
        while instr is not None and body.contains(instr.getAddress()):
            offset = int(instr.getAddress().getOffset())
            lines.append(f"  0x{offset:08x}:  {instr}")
            instr = instr.getNext()
        return "\n".join(lines)

    def _disassemble_range(self, start: int, end: int) -> list[str]:
        """Get disassembly text for instructions in an address range."""
        listing = self.program.getListing()
        start_addr = self._to_addr(start)
        end_addr = self._to_addr(end)
        instructions: list[str] = []
        instr = listing.getInstructionAt(start_addr)
        while instr is not None and instr.getAddress().compareTo(end_addr) <= 0:
            instructions.append(str(instr))
            instr = instr.getNext()
        return instructions

    @staticmethod
    def _classify_flow(flow_type: object) -> BlockEdgeType:
        """Map a Ghidra FlowType to our BlockEdgeType enum."""
        name = str(flow_type)
        if "FALL_THROUGH" in name:
            return BlockEdgeType.FALL_THROUGH
        if "UNCONDITIONAL_JUMP" in name or "CONDITIONAL_JUMP" in name:
            return BlockEdgeType.BRANCH
        if "CALL" in name:
            return BlockEdgeType.CALL
        if "COMPUTED" in name or "INDIRECT" in name:
            return BlockEdgeType.COMPUTED
        return BlockEdgeType.UNKNOWN

    def _to_addr(self, addr: int) -> Any:
        """Convert an integer offset to a Ghidra Address."""
        return self.program.getAddressFactory().getDefaultAddressSpace().getAddress(addr)

    def _get_function_at(self, addr: int) -> Any:
        """Get Ghidra Function object at an address."""
        func = self.program.getFunctionManager().getFunctionAt(self._to_addr(addr))
        if func is None:
            raise GhidraClientError(f"No function found at 0x{addr:08x}")
        return func
