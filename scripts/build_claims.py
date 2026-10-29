"""Attach each claim to an exact span of the CSV full text.

Annotation judgments are listed below. The script only checks that the
evidence span occurs in publications_tdm.csv and writes CLAIMS.csv.
"""

from __future__ import annotations

import csv
import io
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/zenodo/publications_tdm.csv"
OUT_PATH = ROOT / "data" / "CLAIMS.csv"

# doi, label, statement, anchor phrase (whitespace-flexible, case-insensitive)
CLAIMS = [
    (
        "10.3391/ai.2021.16.1.07",
        "supported",
        "In Lake Texoma, densities of zebra mussel veligers and post-veligers were lower at western sites with lower water clarity, higher salinity, and higher productivity.",
        "densities of both life stages were lower at western sites with lower water clarity, higher salinity, and higher productivity",
    ),
    (
        "10.3391/ai.2021.16.1.07",
        "unclear",
        "Flow velocity in Lake Texoma controls zebra mussel propagule pressure and establishment.",
        "Further studies are needed within Lake Texoma to directly measure flow",
    ),
    (
        "10.3391/bir.2022.11.4.21",
        "supported",
        "Quagga mussel was recorded for the first time in Italy, in Lake Garda, and identified by morphology and the mitochondrial COI gene.",
        "Here, we report the first record of quagga mussel, Dreissena bugensis",
    ),
    (
        "10.3391/bir.2022.11.4.21",
        "unclear",
        "The sampled individuals are enough to date when quagga mussel colonized Lake Garda.",
        "a reliable estimate of the colonization timing will require further observations",
    ),
    (
        "10.1007/s10530-021-02644-y",
        "supported",
        "Combined crayfish-tissue and water samples detected Aphanomyces astaci DNA above the limit of detection in invasive crayfish populations.",
        "Combined, the two sampling methods revealed A. astaci presence",
    ),
    (
        "10.1007/s10530-021-02644-y",
        "not supported",
        "Larger crayfish yielded higher detection rates of Aphanomyces astaci.",
        "there was no general increase in detection rate with crayfish size",
    ),
    (
        "10.1186/s12862-015-0296-2",
        "supported",
        "Melanoides tuberculata reached Lakes Malawi and Tanganyika, the Congo River, Nigeria and Cameroon in several independent invasions from Asia.",
        "identify several independent Asian invasions in Lakes Malawi and Tanganyika",
    ),
    (
        "10.4102/abc.v47i2.2177",
        "supported",
        "A FISK score of 14 separated freshwater fishes that could become invasive in South Africa from those unlikely to do so.",
        "A FISK score of 14 was identified as the threshold",
    ),
    (
        "10.4102/abc.v47i2.2177",
        "supported",
        "Of three fishes proposed for aquaculture, Silurus glanis had a high risk of becoming invasive in South Africa and Oncorhynchus tshawytscha was unlikely to be invasive.",
        "Silurus glanis had a high risk of becoming invasive in South Africa",
    ),
    (
        "10.3390/d12020082",
        "not supported",
        "Invasive watermilfoil reduced standing crops and gross primary production of other littoral primary producers.",
        "not significantly different between paired study plots",
    ),
    (
        "10.3390/d12020082",
        "unclear",
        "The missing difference between paired plots was caused by invasive watermilfoil rather than by total macrophyte biomass.",
        "it is unclear if this pattern was driven by differences in IWM",
    ),
    (
        "10.1111/ddi.13501",
        "supported",
        "Reported global costs of invasive freshwater bivalves summed to 63.7 billion 2017 US dollars from 1980 to 2020.",
        "Cumulative total global costs of invasive macrofouling bivalves were $ 63.7 billion",
    ),
    (
        "10.1002/ece3.9493",
        "supported",
        "Freshwater fish species richness increased with the degree of invasion in the studied Italian sites.",
        "positive relationship between species richness and invasion degree",
    ),
    (
        "10.1002/ece3.9493",
        "not supported",
        "Native and non-native freshwater fishes differed in their contribution to beta diversity.",
        "Italian native and exotic species did not differ in their SCBD values",
    ),
    (
        "10.1111/eva.13469",
        "supported",
        "Among the three fishes, population genetic differentiation was strongest in three-spined stickleback and weakest in invasive topmouth gudgeon.",
        "strongest population genetic differentiation in three-spined stickleback",
    ),
    (
        "10.3389/fpls.2019.00647",
        "supported",
        "Ludwigia hexapetala had higher photosynthetic efficiency and tolerated higher light than Nelumbo nucifera and the native macrophytes.",
        "L. hexapetala to have higher photosynthetic efficiency",
    ),
    (
        "10.3389/fpls.2019.00647",
        "not supported",
        "The photophysiology of invasive Nelumbo differed from that of the native macrophytes.",
        "photophysiological properties of Nelumbo did not differ from those of the local macrophytes",
    ),
    (
        "10.1111/geb.13051",
        "not supported",
        "Temperature had a significant effect on invasion of every freshwater group in the English dataset.",
        "temperature surprisingly had no significant effect on any group",
    ),
    (
        "10.1111/geb.13051",
        "supported",
        "Connectivity to human activity, measured along the hydrological network, explained freshwater invasion better than simple proxies of human pressure.",
        "Our findings support the hypothesis that measures of human activity",
    ),
    (
        "10.1007/s10750-020-04318-7",
        "supported",
        "After the third Elodea collapse, submerged macrophytes did not recover and Littoistenjärvi stayed hypertrophic with cyanobacterial blooms.",
        "the submerged macrophytes did not recover",
    ),
    (
        "10.3989/ajbm.2623",
        "supported",
        "Only Ficus benjamina, among 388 non-native woody plants in Spanish urban parks, had no potential to invade.",
        "Only Ficus benjamina has no potential to invade",
    ),
    (
        "10.1093/aobpla/plab042",
        "not supported",
        "Invasion of restored vernal pools shifted soil C:N and microbial biomass relative to natural reference pools.",
        "soil C:N ratios and microbial biomass did not differ between invaded and naturally occurring reference pools",
    ),
    (
        "10.1002/ecs2.2097",
        "not supported",
        "Green tea litter decomposed faster in plots invaded by Impatiens glandulifera than in control plots.",
        "green tea decomposition did not differ significantly between invaded and control plots",
    ),
    (
        "10.3390/f13020273",
        "supported",
        "National provenance recommendations exist for several non-native forest trees planted in Europe, including Douglas fir.",
        "official and non-official national recommendations, based on provenance research results, have been elaborated",
    ),
    (
        "10.3389/fevo.2022.1080599",
        "supported",
        "Brachycaudus helichrysi and Aulacorthum solani were the two most frequent aphids on both Canadian goldenrod and the crop plants.",
        "The two most frequent aphid species, found both on Canadian goldenrod and crop plants",
    ),
    (
        "10.3390/insects13100877",
        "supported",
        "Aculodes marcelli from cheatgrass is a different species from A. altamurgiensis, with 17.7 percent COI divergence.",
        "significant levels of genetic variation (17.7%) and supported the results from the morphometric analysis",
    ),
    (
        "10.3897/italianbotanist.11.68063",
        "supported",
        "The contribution reports new records, confirmations, exclusions and status changes of vascular plants alien to Italy.",
        "new data concerning the distribution of vascular flora alien to Italy",
    ),
    (
        "10.3391/mbi.2017.8.3.17",
        "supported",
        "Pumpkinseed was introduced into an English angling lake as a contaminant of a native aquatic-plant consignment.",
        "introduction of pumpkinseed Lepomis gibbosus, a North American sunfish, into an angling lake as a contaminant",
    ),
    (
        "10.1038/s41467-018-06995-4",
        "supported",
        "In six Central European habitats, naturalized non-invasive plants were functionally similar to native plants, while invasive plants occupied the edge of trait space.",
        "naturalized non-invasive species are functionally similar to native species",
    ),
    (
        "10.3897/neobiota.71.69716",
        "supported",
        "Proximity of riparian sites to roads, railway lines and housing was not related to the level of plant invasion.",
        "housing areas was not related to the invasion level",
    ),
    (
        "10.7717/peerj.6479",
        "supported",
        "Hotspots of potential invasive-plant expansion under climate change include Northern Europe, the UK, South America, North America, southwest China and New Zealand.",
        "Hotspots with a potential risk of IPS",
    ),
    (
        "10.3390/plants11152059",
        "supported",
        "Seed germination of Erigeron bonariensis and E. sumatrensis was reduced more by NaCl than by polyethylene glycol.",
        "Seed germination was mostly affected by NaCl, but less by PEG",
    ),
    (
        "10.3391/ai.2022.17.1.01",
        "supported",
        "A Vaucheria turf near Sylt expanded over about 180 ha within three years and trapped fine sediment.",
        "dense turfs of green filaments rapidly expanded over an area of 180 ha within 3 years",
    ),
    (
        "10.3391/bir.2022.11.1.02",
        "supported",
        "The marine diatom Licmophora normaniana was found in an inland mineral spring in the Siwa Oasis, Egypt.",
        "first known occurrence in an inland mineral spring in the Siwa Oasis",
    ),
    (
        "10.3391/bir.2022.11.1.02",
        "unclear",
        "Licmophora normaniana in the Siwa spring has been shown to alter the local diatom community.",
        "We further hypothesize that the primary negative consequence",
    ),
    (
        "10.1007/s10530-022-02839-x",
        "supported",
        "Idotea balthica reduced activity when exposed to cues of native perch.",
        "the isopods lowered their activity when exposed to the fish cue",
    ),
    (
        "10.1007/s10530-022-02839-x",
        "not supported",
        "Idotea balthica showed anti-predator behaviour when exposed to cues of the invasive crab Rhithropanopeus harrisii.",
        "crab cue did not induce anti-predator behaviour",
    ),
    (
        "10.3389/fmars.2022.983685",
        "supported",
        "Sea surface temperature in northern Japan and the East China Sea rose by about 0.25 to 0.4 degrees Celsius per decade from 1978 to 2018.",
        "SST in northern Japan and the East China Sea indeed increased broadly by 0.25-0.4",
    ),
    (
        "10.1371/journal.pone.0154201",
        "supported",
        "Didemnum perlucidum in the Swan River estuary was concentrated near infrastructure, especially mooring buoys.",
        "D. perlucidum was largely present near areas of infrastructure, particularly mooring buoys",
    ),
    (
        "10.1371/journal.pone.0154201",
        "unclear",
        "The extent of interaction between Didemnum perlucidum and native sessile invertebrates is established.",
        "Further research is needed to determine the extent of interaction",
    ),
    (
        "10.1098/rsos.150039",
        "supported",
        "Hindcast ballast-water discharge into Australian waters more than doubled between 1999 and 2012.",
        "total hindcast ballast water discharge into Australian waters more than doubled",
    ),
    (
        "10.1098/rsos.150039",
        "not supported",
        "Multiple regression effectively modelled ballast-water discharge from bulk carriers.",
        "Ballast discharge from bulkers was not effectively modelled",
    ),
    (
        "10.3989/scimar.2011.75n2301",
        "supported",
        "Mya arenaria reproduced in the Tagus estuary, as shown by recruited juveniles of about 2 mm throughout the study.",
        "the clam is capable of reproducing in the new habitat",
    ),
    (
        "10.3989/scimar.2011.75n2301",
        "supported",
        "Juvenile Mya arenaria were more abundant in the mid-intertidal, while juvenile Scrobicularia plana were mainly in the upper intertidal.",
        "Juvenile specimens of Mya arenaria were more abundant in the mid-intertidal zone",
    ),
    (
        "10.1038/s41598-020-77742-3",
        "supported",
        "Tritia neritea in its native range showed strong genetic structure even at local scales within the Adriatic and the Black Sea.",
        "genetically divergent populations even at local scales within small seas such as the Adriatic Sea",
    ),
    (
        "10.1038/s41598-020-77742-3",
        "unclear",
        "The origin of the unsampled native Tritia neritea haplotypes on the southeastern and far eastern Mediterranean coasts is resolved.",
        "Further sampling on the South Eastern and Far Eastern Mediterranean coasts is warranted",
    ),
    (
        "10.1093/aobpla/plaa059",
        "supported",
        "Red oaks in soil with lower nitrogen concentrations experienced more gypsy-moth herbivory than trees in soil with higher nitrogen.",
        "trees in soil with lower N concentrations experienced more herbivory",
    ),
    (
        "10.1093/aobpla/plaa059",
        "not supported",
        "Within red oak trees, heavier gypsy-moth herbivory occurred on leaves with higher nitrogen concentrations.",
        "no significant relationship between the extent of leaf herbivory and the N concentrations within leaves",
    ),
    (
        "10.1186/s12862-018-1336-5",
        "supported",
        "The invasion of Nylanderia fulva into the United States passed through a bottleneck that lowered genetic diversity by about 60 percent.",
        "genetic bottleneck during its invasion lowering its genetic diversity by 60%",
    ),
    (
        "10.1002/ece3.5217",
        "supported",
        "Small hive beetles completed a life cycle in nests of the solitary bee Megachile rotundata, and flowers served as alternative food.",
        "SHBs can complete an entire life cycle in association with nests of solitary bees",
    ),
    (
        "10.1111/eva.12834",
        "supported",
        "Aedes aegypti in Cape Verde showed no kdr mutations associated with pyrethroid resistance in the sampled mosquitoes.",
        "No kdr mutations associated with pyrethroid resistance were detected",
    ),
    (
        "10.3390/insects11080512",
        "supported",
        "Fall armyworm infestation and damage in central Mozambique were higher in the dry season than in the rainy season.",
        "increased infestation, damages, and population density of FAW in the dry season",
    ),
    (
        "10.3897/neobiota.85.103904",
        "not supported",
        "Bostrichid beetle richness and abundance differed between Spanish port areas and the surrounding natural areas.",
        "did not differ between the two habitats for Bostrichidae",
    ),
    (
        "10.7717/peerj.8103",
        "supported",
        "Forest habitat shaped the fungal communities of both ambrosia beetles, with a stronger effect on the exotic Xylosandrus germanus.",
        "forest habitat shaped the community of fungi associated with both species, but the effect was stronger for the exotic",
    ),
    (
        "10.1371/journal.pone.0181397",
        "supported",
        "Adding irrigation to the CLIMEX model changed the predicted potential distribution of Vespula germanica in dry regions.",
        "When irrigation is applied, dry stress is",
    ),
    (
        "10.3989/ajbm.2452",
        "supported",
        "MaxEnt models for 20 non-native aquatic macrophytes in Iberia discriminated suitable from unsuitable areas with high accuracy.",
        "The accuracy scores of models ranged between 0.918 and 0.981",
    ),
    (
        "10.4102/abc.v49i1.2356",
        "supported",
        "Most respondents in Zambia reported that Tithonia diversifolia reduces native vegetation, access, water, crop yields and animal health.",
        "T. diversifolia has negative impacts on native vegetation, mobility or access, water availability, crop yields and animal health",
    ),
    (
        "10.3390/d15040496",
        "supported",
        "In the fitted two-node model, oak processionary moth infestation in Richmond Park largely drove infestation in neighbouring Bushy Park.",
        "the infestation within Richmond Park has largely driven the infestation within Bushy Park",
    ),
    (
        "10.1111/ddi.13489",
        "supported",
        "For all five modelled hammerhead flatworms, climate variables outweighed soil variables, led by precipitation of the driest month and minimum temperature of the coldest month.",
        "Precipitation of the driest month (BIO 14) and minimum temperature of the coldest month (BIO 6)",
    ),
    (
        "10.1111/ddi.13489",
        "unclear",
        "Soil characteristics do not matter for where the five hammerhead flatworms can occur.",
        "we cannot exclude that at a finer scale, soil characteristics",
    ),
    (
        "10.1002/ecs2.2987",
        "supported",
        "Prosopis juliflora in Eastern Africa is still early in its invasion and is likely to expand if it is not controlled.",
        "P. juliflora invasion in Eastern Africa is still at an early stage",
    ),
    (
        "10.1002/ecs2.2987",
        "supported",
        "Lantana camara in Eastern Africa is closer to a pseudo-stable equilibrium than Prosopis juliflora.",
        "L. camara, by contrast, appears to be approaching a pseudo-stable equilibrium",
    ),
    (
        "10.3390/f9030117",
        "supported",
        "Pine plantations and pine invasion in the Patagonian shrubby steppe increased the amount and continuity of fine fuels.",
        "pine plantations and their subsequent invasion in the Patagonian shrubby steppe produced sharp changes in fine fuel",
    ),
    (
        "10.3390/f9030117",
        "not supported",
        "Low and intermediate pine invasions differed in the amount and vertical distribution of fine fuels.",
        "no significant differences in the amount of fine fuels across the vertical distribution",
    ),
    (
        "10.1111/1365-2664.14287",
        "not supported",
        "In Norway, alien species introduced by escape, as contaminants or as stowaways differed in their area of occupancy.",
        "did not differ in their area of occupancy",
    ),
    (
        "10.3390/su13147541",
        "supported",
        "Active planting of Nothofagus in abandoned beaver ponds in Tierra del Fuego reached average survival above 30 to 40 percent after three years.",
        "survival rates >30–40% in average",
    ),
    (
        "10.3390/su13147541",
        "unclear",
        "The same planting treatment worked equally well across beaver meadows.",
        "the variation in beaver meadows greatly affected",
    ),
]


def load_texts() -> dict[str, str]:
    raw = CSV_PATH.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    texts = {}
    for rec in csv.DictReader(io.StringIO(raw)):
        texts[rec["DOI"].strip()] = rec.get("Full-text") or ""
    return texts


_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def _letters(text: str) -> tuple[str, list[int]]:
    chars = []
    index = []
    for position, char in enumerate(text):
        if char.isspace() or char in "-‐‑–—":
            continue
        expanded = _LIGATURES.get(char, char.lower())
        for piece in expanded:
            chars.append(piece)
            index.append(position)
    return "".join(chars), index


def _is_abbrev(text: str, period: int) -> bool:
    window = text[max(0, period - 8):period].lower()
    if window.endswith(("fig", "et al", "i.e", "e.g")):
        return True
    after = text[period + 1] if period + 1 < len(text) else ""
    before = text[period - 1] if period > 0 else ""
    if after.isdigit() and (before.isdigit() or before.isspace() or before in "−-="):
        return True
    return False


def _sentence_start(text: str, match_start: int) -> int:
    cursor = match_start
    floor = max(0, match_start - 320)
    while cursor > floor:
        period = text.rfind(".", floor, cursor)
        if period == -1:
            break
        if _is_abbrev(text, period):
            cursor = period
            continue
        return period + 1
    return max(0, match_start - 40)


def _sentence_end(text: str, match_end: int) -> int:
    cursor = match_end
    ceiling = min(len(text), match_end + 420)
    while cursor < ceiling:
        period = text.find(".", cursor, ceiling)
        if period == -1:
            return min(len(text), match_end + 220)
        if _is_abbrev(text, period):
            cursor = period + 1
            continue
        return period + 1
    return min(len(text), match_end + 220)


def find_span(text: str, anchor: str) -> str:
    stream, index = _letters(text)
    needle, _ = _letters(anchor)
    position = stream.find(needle)
    if position < 0:
        raise LookupError(anchor)
    match_start = index[position]
    match_end = index[position + len(needle) - 1] + 1
    start = _sentence_start(text, match_start)
    end = _sentence_end(text, match_end)
    span = text[start:end].strip()
    while span and not span[0].isalpha():
        span = span[1:].lstrip()
    for heading in ("DISCUSSION", "RESULTS", "ABSTRACT", "INTRODUCTION", "CONCLUSION"):
        if span.startswith(heading) and len(span) > len(heading) and span[len(heading)].isupper():
            span = span[len(heading):]
            break
    span_letters, _ = _letters(span)
    if needle not in span_letters:
        raise LookupError("span missed anchor: " + anchor)
    if span not in text:
        raise LookupError("span is not an exact substring")
    return span


def main() -> None:
    texts = load_texts()
    rows = []
    per_doi: dict[str, int] = {}
    missing = []
    for doi, label, statement, anchor in CLAIMS:
        per_doi[doi] = per_doi.get(doi, 0) + 1
        if per_doi[doi] > 2:
            raise SystemExit(f"more than two claims for {doi}")
        if doi not in texts:
            raise SystemExit(f"missing full text {doi}")
        try:
            evidence = find_span(texts[doi], anchor)
        except LookupError:
            missing.append((doi, anchor))
            continue
        rows.append(
            {
                "claim_id": f"c{len(rows)+1:03d}",
                "doi": doi,
                "label": label,
                "statement": statement,
                "evidence": evidence,
            }
        )
    if missing:
        for doi, anchor in missing:
            print("MISSING", doi, "|", anchor)
        raise SystemExit(len(missing))
    with OUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["claim_id", "doi", "label", "statement", "evidence"],
        )
        writer.writeheader()
        writer.writerows(rows)
    from collections import Counter

    print("claims", len(rows), "papers", len(per_doi))
    print(Counter(row["label"] for row in rows))


if __name__ == "__main__":
    main()
