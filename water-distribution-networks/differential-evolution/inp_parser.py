from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


_SECTION_HEADER_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")


@dataclass(slots=True)
class InpRecord:
    line_number: int
    raw: str
    content: str
    tokens: List[str]
    inline_comment: Optional[str] = None


@dataclass(slots=True)
class InpSection:
    name: str
    header_line: int
    records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class SectionSummary:
    name: str
    entry_count: int
    invalid_count: int
    fields: Dict[str, str]


@dataclass(slots=True)
class TitleEntry:
    text: str


@dataclass(slots=True)
class JunctionEntry:
    id: str
    elevation: float
    demand: Optional[float]
    pattern: Optional[str]


@dataclass(slots=True)
class ReservoirEntry:
    id: str
    head: float
    pattern: Optional[str]


@dataclass(slots=True)
class TankEntry:
    id: str
    elevation: float
    init_level: float
    min_level: float
    max_level: float
    diameter: float
    min_vol: Optional[float]
    vol_curve: Optional[str]


@dataclass(slots=True)
class PipeEntry:
    id: str
    node1: str
    node2: str
    length: float
    diameter: float
    roughness: float
    minor_loss: Optional[float]
    status: Optional[str]


@dataclass(slots=True)
class PumpEntry:
    id: str
    node1: str
    node2: str
    parameters: List[str]


@dataclass(slots=True)
class ValveEntry:
    id: str
    node1: str
    node2: str
    diameter: float
    valve_type: str
    setting: str
    minor_loss: Optional[float]


@dataclass(slots=True)
class DemandEntry:
    junction: str
    demand: float
    pattern: Optional[str]
    category: Optional[str]


@dataclass(slots=True)
class PatternEntry:
    id: str
    values: List[float]


@dataclass(slots=True)
class CurveEntry:
    id: str
    x: float
    y: float


@dataclass(slots=True)
class StatusEntry:
    id: str
    status_or_setting: str


@dataclass(slots=True)
class EmitterEntry:
    junction: str
    coefficient: float


@dataclass(slots=True)
class QualityEntry:
    node: str
    init_quality: float


@dataclass(slots=True)
class SourceEntry:
    node: str
    source_type: str
    quality: float
    pattern: Optional[str]


@dataclass(slots=True)
class CoordinateEntry:
    node: str
    x: float
    y: float


@dataclass(slots=True)
class VertexEntry:
    link: str
    x: float
    y: float


@dataclass(slots=True)
class KeyValueEntry:
    key: str
    value: str


@dataclass(slots=True)
class RawEntry:
    raw: str
    tokens: List[str]


@dataclass(slots=True)
class TitleSection:
    entries: List[TitleEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class JunctionsSection:
    entries: List[JunctionEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class ReservoirsSection:
    entries: List[ReservoirEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class TanksSection:
    entries: List[TankEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class PipesSection:
    entries: List[PipeEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class PumpsSection:
    entries: List[PumpEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class ValvesSection:
    entries: List[ValveEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class DemandsSection:
    entries: List[DemandEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class PatternsSection:
    entries: List[PatternEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class CurvesSection:
    entries: List[CurveEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class StatusSection:
    entries: List[StatusEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class EmittersSection:
    entries: List[EmitterEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class QualitySection:
    entries: List[QualityEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class SourcesSection:
    entries: List[SourceEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class CoordinatesSection:
    entries: List[CoordinateEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class VerticesSection:
    entries: List[VertexEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class TimesSection:
    entries: List[KeyValueEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class OptionsSection:
    entries: List[KeyValueEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class ReportSection:
    entries: List[KeyValueEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class EnergySection:
    entries: List[KeyValueEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class KeyValueSection:
    entries: List[KeyValueEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class RawSection:
    entries: List[RawEntry] = field(default_factory=list)
    invalid_records: List[InpRecord] = field(default_factory=list)


@dataclass(slots=True)
class ParsedInpFile:
    path: Path
    sections: Dict[str, List[InpSection]] = field(default_factory=dict)
    section_order: List[str] = field(default_factory=list)
    preamble: List[InpRecord] = field(default_factory=list)
    title: Optional[TitleSection] = None
    junctions: Optional[JunctionsSection] = None
    reservoirs: Optional[ReservoirsSection] = None
    tanks: Optional[TanksSection] = None
    pipes: Optional[PipesSection] = None
    pumps: Optional[PumpsSection] = None
    valves: Optional[ValvesSection] = None
    demands: Optional[DemandsSection] = None
    patterns: Optional[PatternsSection] = None
    curves: Optional[CurvesSection] = None
    status: Optional[StatusSection] = None
    emitters: Optional[EmittersSection] = None
    quality: Optional[QualitySection] = None
    sources: Optional[SourcesSection] = None
    coordinates: Optional[CoordinatesSection] = None
    vertices: Optional[VerticesSection] = None
    times: Optional[TimesSection] = None
    options: Optional[OptionsSection] = None
    report: Optional[ReportSection] = None
    energy: Optional[EnergySection] = None
    raw_sections: Dict[str, RawSection] = field(default_factory=dict)

    def section_summaries(self) -> List[SectionSummary]:
        result: List[SectionSummary] = []
        mapping: List[tuple[str, Any]] = [
            ("TITLE", self.title),
            ("JUNCTIONS", self.junctions),
            ("RESERVOIRS", self.reservoirs),
            ("TANKS", self.tanks),
            ("PIPES", self.pipes),
            ("PUMPS", self.pumps),
            ("VALVES", self.valves),
            ("DEMANDS", self.demands),
            ("PATTERNS", self.patterns),
            ("CURVES", self.curves),
            ("STATUS", self.status),
            ("EMITTERS", self.emitters),
            ("QUALITY", self.quality),
            ("SOURCES", self.sources),
            ("COORDINATES", self.coordinates),
            ("VERTICES", self.vertices),
            ("TIMES", self.times),
            ("OPTIONS", self.options),
            ("REPORT", self.report),
            ("ENERGY", self.energy),
        ]
        for section_name, section_obj in mapping:
            if section_obj is None:
                continue
            entry_fields = _entry_type_fields(section_obj)
            result.append(
                SectionSummary(
                    name=section_name,
                    entry_count=len(section_obj.entries),
                    invalid_count=len(section_obj.invalid_records),
                    fields=entry_fields,
                )
            )
        for section_name, section_obj in sorted(self.raw_sections.items()):
            result.append(
                SectionSummary(
                    name=section_name,
                    entry_count=len(section_obj.entries),
                    invalid_count=len(section_obj.invalid_records),
                    fields=_entry_type_fields(section_obj),
                )
            )
        return result


def _entry_type_fields(section_obj: Any) -> Dict[str, str]:
    if section_obj.entries:
        entry_type = type(section_obj.entries[0])
    else:
        section_to_entry_type: Dict[type, type] = {
            TitleSection: TitleEntry,
            JunctionsSection: JunctionEntry,
            ReservoirsSection: ReservoirEntry,
            TanksSection: TankEntry,
            PipesSection: PipeEntry,
            PumpsSection: PumpEntry,
            ValvesSection: ValveEntry,
            DemandsSection: DemandEntry,
            PatternsSection: PatternEntry,
            CurvesSection: CurveEntry,
            StatusSection: StatusEntry,
            EmittersSection: EmitterEntry,
            QualitySection: QualityEntry,
            SourcesSection: SourceEntry,
            CoordinatesSection: CoordinateEntry,
            VerticesSection: VertexEntry,
            TimesSection: KeyValueEntry,
            OptionsSection: KeyValueEntry,
            ReportSection: KeyValueEntry,
            EnergySection: KeyValueEntry,
            KeyValueSection: KeyValueEntry,
            RawSection: RawEntry,
        }
        entry_type = section_to_entry_type.get(type(section_obj))
        if entry_type is None:
            return {}
    return {f.name: str(f.type).replace("typing.", "") for f in fields(entry_type)}


class InpFileParser:
    SECTION_ALIASES = {"COORDS": "COORDINATES"}

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def parse(self) -> ParsedInpFile:
        text = self.file_path.read_text(encoding="utf-8", errors="replace")
        parsed = ParsedInpFile(path=self.file_path)
        current_section: Optional[InpSection] = None

        for idx, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            header_match = _SECTION_HEADER_RE.match(line)
            if header_match:
                section_name = header_match.group(1).strip().upper()
                current_section = InpSection(name=section_name, header_line=idx)
                parsed.sections.setdefault(section_name, []).append(current_section)
                parsed.section_order.append(section_name)
                continue

            if not line or line.startswith(";"):
                continue

            content, inline_comment = self._split_inline_comment(raw_line)
            cleaned_content = content.strip()
            if not cleaned_content:
                continue

            record = InpRecord(
                line_number=idx,
                raw=raw_line,
                content=cleaned_content,
                tokens=cleaned_content.split(),
                inline_comment=inline_comment,
            )
            if current_section is None:
                parsed.preamble.append(record)
            else:
                current_section.records.append(record)

        self._build_typed_sections(parsed)
        return parsed

    @staticmethod
    def _split_inline_comment(line: str) -> tuple[str, Optional[str]]:
        if ";" not in line:
            return line, None
        content, comment = line.split(";", 1)
        stripped_comment = comment.strip()
        return content, stripped_comment if stripped_comment else None

    def _build_typed_sections(self, parsed: ParsedInpFile) -> None:
        for raw_name, raw_sections in parsed.sections.items():
            name = self.SECTION_ALIASES.get(raw_name, raw_name)
            records = [record for section in raw_sections for record in section.records]
            if not records:
                continue

            if name == "TITLE":
                parsed.title = self._parse_title(records)
            elif name == "JUNCTIONS":
                parsed.junctions = self._parse_junctions(records)
            elif name == "RESERVOIRS":
                parsed.reservoirs = self._parse_reservoirs(records)
            elif name == "TANKS":
                parsed.tanks = self._parse_tanks(records)
            elif name == "PIPES":
                parsed.pipes = self._parse_pipes(records)
            elif name == "PUMPS":
                parsed.pumps = self._parse_pumps(records)
            elif name == "VALVES":
                parsed.valves = self._parse_valves(records)
            elif name == "DEMANDS":
                parsed.demands = self._parse_demands(records)
            elif name == "PATTERNS":
                parsed.patterns = self._parse_patterns(records)
            elif name == "CURVES":
                parsed.curves = self._parse_curves(records)
            elif name == "STATUS":
                parsed.status = self._parse_status(records)
            elif name == "EMITTERS":
                parsed.emitters = self._parse_emitters(records)
            elif name == "QUALITY":
                parsed.quality = self._parse_quality(records)
            elif name == "SOURCES":
                parsed.sources = self._parse_sources(records)
            elif name == "COORDINATES":
                parsed.coordinates = self._parse_coordinates(records)
            elif name == "VERTICES":
                parsed.vertices = self._parse_vertices(records)
            elif name == "TIMES":
                parsed.times = self._parse_key_values(records)
            elif name == "OPTIONS":
                parsed.options = self._parse_key_values(records)
            elif name == "REPORT":
                parsed.report = self._parse_key_values(records)
            elif name == "ENERGY":
                parsed.energy = self._parse_key_values(records)
            else:
                parsed.raw_sections[name] = self._parse_raw(records)

    def _parse_title(self, records: List[InpRecord]) -> TitleSection:
        section = TitleSection()
        for record in records:
            section.entries.append(TitleEntry(text=record.content))
        return section

    def _parse_junctions(self, records: List[InpRecord]) -> JunctionsSection:
        section = JunctionsSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    JunctionEntry(
                        id=tokens[0],
                        elevation=float(tokens[1]),
                        demand=float(tokens[2]) if len(tokens) > 2 else None,
                        pattern=tokens[3] if len(tokens) > 3 else None,
                    )
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_reservoirs(self, records: List[InpRecord]) -> ReservoirsSection:
        section = ReservoirsSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    ReservoirEntry(
                        id=tokens[0],
                        head=float(tokens[1]),
                        pattern=tokens[2] if len(tokens) > 2 else None,
                    )
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_tanks(self, records: List[InpRecord]) -> TanksSection:
        section = TanksSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    TankEntry(
                        id=tokens[0],
                        elevation=float(tokens[1]),
                        init_level=float(tokens[2]),
                        min_level=float(tokens[3]),
                        max_level=float(tokens[4]),
                        diameter=float(tokens[5]),
                        min_vol=float(tokens[6]) if len(tokens) > 6 else None,
                        vol_curve=tokens[7] if len(tokens) > 7 else None,
                    )
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_pipes(self, records: List[InpRecord]) -> PipesSection:
        section = PipesSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    PipeEntry(
                        id=tokens[0],
                        node1=tokens[1],
                        node2=tokens[2],
                        length=float(tokens[3]),
                        diameter=float(tokens[4]),
                        roughness=float(tokens[5]),
                        minor_loss=float(tokens[6]) if len(tokens) > 6 else None,
                        status=tokens[7] if len(tokens) > 7 else None,
                    )
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_pumps(self, records: List[InpRecord]) -> PumpsSection:
        section = PumpsSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    PumpEntry(
                        id=tokens[0],
                        node1=tokens[1],
                        node2=tokens[2],
                        parameters=tokens[3:],
                    )
                )
            except IndexError:
                section.invalid_records.append(record)
        return section

    def _parse_valves(self, records: List[InpRecord]) -> ValvesSection:
        section = ValvesSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    ValveEntry(
                        id=tokens[0],
                        node1=tokens[1],
                        node2=tokens[2],
                        diameter=float(tokens[3]),
                        valve_type=tokens[4],
                        setting=tokens[5],
                        minor_loss=float(tokens[6]) if len(tokens) > 6 else None,
                    )
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_demands(self, records: List[InpRecord]) -> DemandsSection:
        section = DemandsSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    DemandEntry(
                        junction=tokens[0],
                        demand=float(tokens[1]),
                        pattern=tokens[2] if len(tokens) > 2 else None,
                        category=tokens[3] if len(tokens) > 3 else None,
                    )
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_patterns(self, records: List[InpRecord]) -> PatternsSection:
        section = PatternsSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    PatternEntry(
                        id=tokens[0],
                        values=[float(value) for value in tokens[1:]],
                    )
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_curves(self, records: List[InpRecord]) -> CurvesSection:
        section = CurvesSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    CurveEntry(id=tokens[0], x=float(tokens[1]), y=float(tokens[2]))
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_status(self, records: List[InpRecord]) -> StatusSection:
        section = StatusSection()
        for record in records:
            tokens = record.tokens
            if len(tokens) < 2:
                section.invalid_records.append(record)
                continue
            section.entries.append(
                StatusEntry(id=tokens[0], status_or_setting=" ".join(tokens[1:]))
            )
        return section

    def _parse_emitters(self, records: List[InpRecord]) -> EmittersSection:
        section = EmittersSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    EmitterEntry(junction=tokens[0], coefficient=float(tokens[1]))
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_quality(self, records: List[InpRecord]) -> QualitySection:
        section = QualitySection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    QualityEntry(node=tokens[0], init_quality=float(tokens[1]))
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_sources(self, records: List[InpRecord]) -> SourcesSection:
        section = SourcesSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    SourceEntry(
                        node=tokens[0],
                        source_type=tokens[1],
                        quality=float(tokens[2]),
                        pattern=tokens[3] if len(tokens) > 3 else None,
                    )
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_coordinates(self, records: List[InpRecord]) -> CoordinatesSection:
        section = CoordinatesSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    CoordinateEntry(node=tokens[0], x=float(tokens[1]), y=float(tokens[2]))
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_vertices(self, records: List[InpRecord]) -> VerticesSection:
        section = VerticesSection()
        for record in records:
            try:
                tokens = record.tokens
                section.entries.append(
                    VertexEntry(link=tokens[0], x=float(tokens[1]), y=float(tokens[2]))
                )
            except (IndexError, ValueError):
                section.invalid_records.append(record)
        return section

    def _parse_key_values(self, records: List[InpRecord]) -> KeyValueSection:
        section = KeyValueSection()
        for record in records:
            tokens = record.tokens
            if len(tokens) < 2:
                section.invalid_records.append(record)
                continue
            section.entries.append(
                KeyValueEntry(key=" ".join(tokens[:-1]), value=tokens[-1])
            )
        return section

    def _parse_raw(self, records: List[InpRecord]) -> RawSection:
        section = RawSection()
        for record in records:
            section.entries.append(RawEntry(raw=record.content, tokens=record.tokens))
        return section

