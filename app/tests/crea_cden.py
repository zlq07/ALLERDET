from alignment import create_fasta_file_without_duplications

myNonAllergenDataset = '../alignments/nonallergens_data_set.fasta'
naAllerTop = '../alignments/reduced_all_nonallergens.fasta'
naAllerHunter = '../alignments/AllerHunter/trainingdata/training.putative_non_allergen.fa'
naPlantProt = '../alignments/unitprot/non/plant_nonallergen.fasta'
naCowsMilkProt = '../alignments/unitprot/non/cowmilk_nonallergen.fasta'
naEggsProt = '../alignments/unitprot/non/eggs_nonallergen.fasta'
naSalmonProt = '../alignments/unitprot/non/salmo-nonallergen.fasta'

# dataset non allergens de uniprot, AllerTop y AllerHunter
create_fasta_file_without_duplications([naPlantProt, naCowsMilkProt, naEggsProt, naSalmonProt
    , naAllerTop, naAllerHunter], myNonAllergenDataset, maxSec=2000)