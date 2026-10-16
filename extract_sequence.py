#!/usr/bin/env python3
"""Extract observed protein sequences from the first PDB/mmCIF model.

Install: python -m pip install biopython
Usage:   python extract_sequence.py structure.cif reference.fasta --out-prefix result
Outputs: result.fasta (one record per chain) and result.matches.tsv (all chain/reference pairs).
Without --out-prefix, outputs are beside the structure, named <stem>.model1.*.
Existing output files are never overwritten.

The first model means the first encountered model, even if its serial number is
not 1. Author chain IDs are retained; blank chain IDs display as <blank> in the
table. Missing coordinate residues are omitted, not reconstructed. RNA/DNA,
water and unrecognized ligands are skipped. Known modified amino acids map to
their parent amino acid; UNK is X, SEC is U and PYL is O. Unrecognized residues
with a protein backbone are retained as X with a warning.

FASTA must contain unaligned protein sequences with unique IDs. Lowercase is
accepted, and one terminal stop (*) is removed. Every extracted chain is
compared with every FASTA record, without assuming matching chain/record names.
Exact match is literal sequence equality, including ambiguous letters.
Identity = identical letters / all global-alignment columns, including gaps.
Ambiguous letters (X/B/Z/J) are literal, not wildcards or proven residue identity.
Global alignment scores: match +2, mismatch -1, gap open -3, gap extend -0.5.
When optimal alignments tie, the first returned alignment determines the metrics.
This is a sequence comparison, not a homology significance test.
"""

import argparse
import csv
import io
from pathlib import Path
from urllib.parse import quote
import warnings

try:
    from Bio import SeqIO
    from Bio.Align import PairwiseAligner
    from Bio.Data.PDBData import protein_letters_3to1_extended
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.PDBExceptions import PDBConstructionException
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
except ImportError:
    raise SystemExit("Biopython is required. Install it with: python -m pip install biopython")


AMINO_ACIDS = {**protein_letters_3to1_extended, "UNK": "X", "SEC": "U", "PYL": "O"}
FASTA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def extract_sequences(path: Path) -> tuple[int, dict[str, str]]:
    """Return the first model's serial number and observed sequence for each protein chain."""
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        parser = PDBParser(PERMISSIVE=False)
    elif suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(auth_chains=True, auth_residues=True)
    else:
        raise ValueError("Structure file must end in .pdb, .cif or .mmcif")
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError("Structure contains no coordinate models")
    sequences = {}
    for chain in model:
        letters = []
        for residue in chain:
            name = residue.get_resname().strip().upper()
            if residue.id[0] == "W":
                continue
            # shortcut: known amino-acid ligands also match this dictionary; use
            # polymer metadata if such ligands must be distinguished from chains.
            letter = AMINO_ACIDS.get(name)
            if letter is None and (
                all(atom in residue for atom in ("N", "CA", "C"))
                or (residue.id[0] == " " and "CA" in residue)
            ):
                letter = "X"
                warnings.warn(f"Unknown protein residue {name} in chain {chain.id!r}, "
                              f"residue {residue.id}: using X", stacklevel=2)
            if letter is not None:
                letters.append(letter)
        if letters:
            sequences[chain.id] = "".join(letters)
    if not sequences:
        raise ValueError("The first model contains no recognized protein residues")
    return model.serial_num, sequences


def read_fasta(path: Path) -> list[SeqRecord]:
    """Read and validate unaligned protein FASTA records."""
    with path.open(encoding="utf-8-sig") as handle:
        first_line = next((line for line in handle if line.strip()), "")
        if not first_line.startswith(">"):
            raise ValueError("FASTA must begin with a >header and contain a sequence")
        handle.seek(0)
        records = list(SeqIO.parse(handle, "fasta"))
    seen = set()
    for record in records:
        if not record.id or record.id in seen:
            raise ValueError(f"Missing or duplicate FASTA ID: {record.id!r}")
        seen.add(record.id)
        sequence = str(record.seq).upper().removesuffix("*")
        invalid = set(sequence) - FASTA_ALPHABET
        if not sequence or invalid:
            raise ValueError(f"FASTA record {record.id!r} is empty or contains unsupported "
                             f"characters: {''.join(sorted(invalid))!r}")
        record.seq = Seq(sequence)
    if not records:
        raise ValueError("FASTA contains no records")
    return records


def compare_sequences(observed: str, reference: str) -> dict:
    """Compare complete sequences using a global alignment with explicit scoring."""
    if not observed or not reference:
        raise ValueError("Cannot align empty sequences")
    aligner = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                              open_gap_score=-3, extend_gap_score=-0.5)
    alignment = aligner.align(observed, reference)[0]
    left, right = alignment[0], alignment[1]
    matches = sum(a == b for a, b in zip(left, right))
    gaps = sum(a == "-" or b == "-" for a, b in zip(left, right))
    return {
        "exact_match": observed == reference,
        "identity_pct": 100 * matches / len(left),
        "matches": matches,
        "mismatches": len(left) - matches - gaps,
        "gap_columns": gaps,
        "alignment_columns": len(left),
        "alignment_score": alignment.score,
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("structure", type=Path, help="Input .pdb, .cif or .mmcif file")
    cli.add_argument("fasta", type=Path, help="Reference protein FASTA file")
    cli.add_argument("--out-prefix", type=Path, help="Output prefix; default: <structure stem>.model1")
    args = cli.parse_args()
    prefix = args.out_prefix or args.structure.with_name(args.structure.stem + ".model1")
    fasta_out, report_out = Path(str(prefix) + ".fasta"), Path(str(prefix) + ".matches.tsv")
    try:
        for output in (fasta_out, report_out):
            if output.exists():
                raise FileExistsError(f"Output already exists: {output}; choose another --out-prefix")
        model_serial, chains = extract_sequences(args.structure)
        references = read_fasta(args.fasta)
        rows = []
        for chain, sequence in chains.items():
            for reference in references:
                row = {"model_serial": model_serial, "chain": chain if chain.strip() else "<blank>",
                       "structure_length": len(sequence), "fasta_id": reference.id,
                       "fasta_length": len(reference.seq)}
                row.update(compare_sequences(sequence, str(reference.seq)))
                row["identity_pct"] = f"{row['identity_pct']:.2f}"
                rows.append(row)
        records = [SeqRecord(Seq(sequence),
                             id=f"model1_chain={quote(chain, safe='')}",
                             description=f"model_serial={model_serial} observed_protein_residues")
                   for chain, sequence in chains.items()]
        fasta_text = io.StringIO()
        SeqIO.write(records, fasta_text, "fasta")
        # Exclusive creation protects existing inputs/outputs even during concurrent runs.
        with fasta_out.open("x", encoding="utf-8") as handle:
            handle.write(fasta_text.getvalue())
        with report_out.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError, KeyError, IndexError, PDBConstructionException) as exc:
        cli.error(str(exc))
    print(f"Extracted {len(chains)} protein chain(s) from the first model (serial {model_serial}).")
    print(f"Exact sequence matches: {sum(row['exact_match'] for row in rows)} / {len(rows)} comparisons.")
    print(f"Sequences: {fasta_out}")
    print(f"Comparison: {report_out}")


if __name__ == "__main__":
    main()
