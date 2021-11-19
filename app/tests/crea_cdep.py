from alignment import splitFastaSeqs
from alignment import create_fasta_file_without_duplications

myAllergenDataset = '../alignments/allergens_data_set.fasta'
aUniprotReviewed = '../alignments/unitprot/uniprot-allergy+OR+atopy+OR+allergen+OR+allergome.fasta'
aAllerTop = '../alignments/reduced_all_allergens.fasta'
aAllerHunter = '../alignments/AllerHunter/trainingdata/training.allergen.fa'

# combina diferentes datasets de alérgenos sin duplicaciones
create_fasta_file_without_duplications([aUniprotReviewed, aAllerTop, aAllerHunter]
    , myAllergenDataset
    , splitFastaSeqs('../alignments/allergens_allertop_1000.fasta')[1], 2000)

