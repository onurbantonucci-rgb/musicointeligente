"""Fontes de Cifras e Documentos Musicais (ChartSources v0.4).

Padroniza a entrada de cifras a partir de múltiplos formatos:
- Arquivos de Texto (.txt) ou texto colado diretamente na interface
- Documentos do Microsoft Word (.docx)
- Imagens de partituras/cifras (.png, .jpg, .jpeg, .webp)

Todas as fontes produzem uma representação intermediária comum (RawChartDocument)
antes de ingressar no ChartParser e convergir para o ChordChart.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os

try:
    import docx
except ImportError:
    docx = None

try:
    import pypdf
except ImportError:
    pypdf = None


@dataclass
class ExtractedToken:
    """Token individual extraído (palavra, acorde ou anotação) com coordenadas espaciais."""
    text: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    confidence: float = 1.0
    is_chord: bool = False
    is_low_confidence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "confidence": round(self.confidence, 2),
            "is_chord": self.is_chord,
            "is_low_confidence": self.is_low_confidence,
        }


@dataclass
class RawChartDocument:
    """Representação intermediária comum de uma cifra antes do parsing sintático."""
    text: str                                           # Texto estruturado alinhado
    tokens: List[ExtractedToken] = field(default_factory=list)
    overall_confidence: float = 1.0
    low_confidence_tokens: List[ExtractedToken] = field(default_factory=list)
    source_type: str = "text"                           # "text", "docx", "image"
    source_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_low_confidence_chords(self) -> bool:
        """Indica se há acordes extraídos com baixa confiança que exigem revisão manual."""
        return len(self.low_confidence_tokens) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
            "overall_confidence": round(self.overall_confidence, 2),
            "low_confidence_tokens": [t.to_dict() for t in self.low_confidence_tokens],
            "source_type": self.source_type,
            "source_path": self.source_path,
            "metadata": dict(self.metadata),
        }


class ChartSource(ABC):
    """Interface abstrata fundamental para qualquer fonte de cifra."""

    @abstractmethod
    def load(self, source_input: Any) -> RawChartDocument:
        """Carrega os dados da fonte e produz um RawChartDocument intermediário."""
        pass


class TextChartSource(ChartSource):
    """Fonte para arquivos de texto puro (.txt) ou texto colado diretamente."""

    def load(self, source_input: str) -> RawChartDocument:
        """Recebe texto bruto ou caminho para arquivo .txt."""
        raw_text = ""
        source_path = ""

        # Verifica se o input é um caminho de arquivo existente
        if isinstance(source_input, str) and os.path.isfile(source_input):
            source_path = os.path.abspath(source_input)
            # Tenta decodificar em UTF-8 com fallback para Latin-1
            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except UnicodeDecodeError:
                with open(source_path, "r", encoding="latin-1") as f:
                    raw_text = f.read()
        else:
            raw_text = str(source_input)

        # Remove caracteres indesejados mantendo quebras de linha e tabulações
        cleaned_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        return RawChartDocument(
            text=cleaned_text,
            source_type="text",
            source_path=source_path,
            overall_confidence=1.0,
            metadata={"lines_count": len(cleaned_text.splitlines())}
        )


class DocxChartSource(ChartSource):
    """Fonte para documentos do Microsoft Word (.docx)."""

    def load(self, source_input: str) -> RawChartDocument:
        """Lê arquivo .docx extraindo parágrafos e tabelas com preservação de estrutura."""
        if not os.path.isfile(source_input):
            raise FileNotFoundError(f"Arquivo DOCX não encontrado: {source_input}")

        if docx is None:
            raise ImportError(
                "Importação de .docx requer o pacote 'python-docx'. "
                "Instale com: pip install python-docx"
            )

        doc = docx.Document(source_input)
        extracted_lines = []
        detected_title = ""

        # 1. Extrai parágrafos na ordem
        for para in doc.paragraphs:
            txt = para.text.strip()
            if not txt:
                continue
            is_heading = para.style and para.style.name and any(h in para.style.name.lower() for h in ["title", "heading"])
            if is_heading and not detected_title:
                detected_title = txt
                extracted_lines.append(f"Título: {txt}")
            else:
                extracted_lines.append(para.text)

        # 2. Extrai tabelas se existirem na cifra
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    extracted_lines.append("    ".join(cells))

        full_text = "\n".join(extracted_lines)

        meta = {"paragraphs_count": len(doc.paragraphs)}
        if detected_title:
            meta["title"] = detected_title

        return RawChartDocument(
            text=full_text,
            source_type="docx",
            source_path=os.path.abspath(source_input),
            overall_confidence=1.0,
            metadata=meta
        )


class ImageChartSource(ChartSource):
    """Fonte para imagens contendo cifras e partituras (.png, .jpg, .jpeg, .webp)."""

    def __init__(self, extractor=None):
        from app.input.image_extractor import ImageChartExtractor
        self._extractor = extractor or ImageChartExtractor()

    def load(self, source_input: Any) -> RawChartDocument:
        """Recebe caminho de arquivo de imagem ou objeto PIL.Image."""
        return self._extractor.extract(source_input)


class PdfChartSource(ChartSource):
    """Fonte para documentos em formato Portable Document Format (.pdf).

    Suporta de forma transparente:
    1. PDFs digitais com camada de texto: extração estruturada preservando alinhamento 2D.
    2. PDFs escaneados / imagens embutidas: extrai as imagens das páginas e
       repassa para o ImageChartExtractor (OCR nativo com projeção espacial).
    """

    def __init__(self, extractor=None):
        self._image_extractor = extractor

    def load(self, source_input: str) -> RawChartDocument:
        """Carrega e extrai conteúdo de arquivo PDF (texto digital ou páginas escaneadas)."""
        if not os.path.isfile(source_input):
            raise FileNotFoundError(f"Arquivo PDF não encontrado: {source_input}")

        if pypdf is None:
            raise ImportError(
                "Importação de .pdf requer o pacote 'pypdf'. "
                "Instale com: pip install pypdf"
            )

        reader = pypdf.PdfReader(source_input)
        page_texts = []
        has_scanned_pages = False
        all_tokens = []
        low_confidence_tokens = []
        overall_conf = 1.0

        for page_idx, page in enumerate(reader.pages):
            # 1. Tentar extração de texto digital preservando layout 2D
            text = ""
            try:
                text = page.extract_text(extraction_mode="layout") or ""
            except Exception:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

            cleaned_page_text = text.strip()
            # 2. Se a página não contiver texto suficiente, verificar se há imagens escaneadas
            if len(cleaned_page_text) < 15 and hasattr(page, "images") and len(page.images) > 0:
                has_scanned_pages = True
                if self._image_extractor is None:
                    from app.input.image_extractor import ImageChartExtractor
                    self._image_extractor = ImageChartExtractor()

                for img_file in page.images:
                    try:
                        import io
                        from PIL import Image
                        pil_img = Image.open(io.BytesIO(img_file.data))
                        img_doc = self._image_extractor.extract(pil_img)
                        page_texts.append(img_doc.text)
                        all_tokens.extend(img_doc.tokens)
                        low_confidence_tokens.extend(img_doc.low_confidence_tokens)
                        overall_conf = min(overall_conf, img_doc.overall_confidence)
                    except Exception:
                        pass
            else:
                page_texts.append(text)

        full_text = "\n\n".join(page_texts).strip()

        return RawChartDocument(
            text=full_text,
            tokens=all_tokens,
            low_confidence_tokens=low_confidence_tokens,
            overall_confidence=round(overall_conf if has_scanned_pages else 1.0, 2),
            source_type="pdf",
            source_path=os.path.abspath(source_input),
            metadata={
                "pages_count": len(reader.pages),
                "is_scanned": has_scanned_pages,
            }
        )
