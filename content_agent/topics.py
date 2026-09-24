"""Áreas de foco do agente de conteúdo do ClinicGPT."""

AREAS = {
    "emagrecimento": {
        "label": "Emagrecimento e Obesidade",
        "termos_busca": [
            "obesity treatment",
            "weight loss pharmacotherapy",
            "GLP-1 receptor agonist weight loss",
            "semaglutide obesity",
            "tirzepatide weight loss",
            "bariatric surgery outcomes",
        ],
    },
    "reposicao_hormonal_feminina": {
        "label": "Reposição Hormonal Feminina",
        "termos_busca": [
            "menopause hormone replacement therapy",
            "estrogen therapy menopause",
            "bioidentical hormone therapy women",
            "testosterone therapy women",
            "perimenopause treatment",
        ],
    },
    "reposicao_hormonal_masculina": {
        "label": "Reposição Hormonal Masculina (TRT)",
        "termos_busca": [
            "testosterone replacement therapy men",
            "male hypogonadism treatment",
            "TRT cardiovascular risk",
            "andropause diagnosis treatment",
        ],
    },
    "longevidade": {
        "label": "Longevidade e Medicina Anti-Aging",
        "termos_busca": [
            "longevity intervention humans",
            "healthspan biomarkers aging",
            "rapamycin aging humans",
            "NAD+ precursor supplementation aging",
            "caloric restriction longevity",
        ],
    },
    "nutrologia": {
        "label": "Nutrologia e Nutrição Clínica",
        "termos_busca": [
            "micronutrient deficiency clinical",
            "vitamin D supplementation outcomes",
            "clinical nutrition metabolic health",
            "protein intake body composition",
        ],
    },
}


def termo_padrao(area: str) -> str:
    """Combina os termos de busca padrão da área em uma única query (OR)."""
    dados = AREAS.get(area)
    if not dados:
        raise ValueError(f"Área desconhecida: {area}. Áreas válidas: {list(AREAS.keys())}")
    return " OR ".join(dados["termos_busca"])
