from alignment import splitFastaSeqs
from alignment import create_fasta_file_without_duplications

myAllergenDataset = '../alignments/allergens_data_set.fasta'
myAlleTestDataset = '../alignments/test_allergens_data_set.fasta'
aAllerTop = '../alignments/reduced_all_allergens.fasta'
aAllerHunter = '../alignments/AllerHunter/trainingdata/training.allergen.fa'
aAllerHunterTest = '../alignments/AllerHunter/testingdata/testing.allergen.fa'
aAllerHunterInd = '../alignments/AllerHunter/independentdata/indp.allergen.fa'
naAllerTop = '../alignments/reduced_all_nonallergens.fasta'
naAllerHunter = '../alignments/AllerHunter/trainingdata/training.putative_non_allergen.fa'
myNonAllergenDataset = '../alignments/nonallergens_data_set.fasta'
myNonAllergenTestDataset = '../alignments/test_nonallergens_data_set.fasta'

# creación del conjunto de test de alérgenos
exclusion=splitFastaSeqs(myAllergenDataset)[1]
create_fasta_file_without_duplications([aAllerTop, aAllerHunter, aAllerHunterInd, aAllerHunterTest], myAlleTestDataset, exclusion, 800)

# creación del conjunto de test de no alérgenos
exclusion=splitFastaSeqs(myNonAllergenDataset)[1]
create_fasta_file_without_duplications([naAllerTop, naAllerHunter], myNonAllergenTestDataset, exclusion, 800)