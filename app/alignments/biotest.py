'''
from Bio import SeqIO

for seq_record in SeqIO.parse("reduced_all_allergens.fasta", "fasta"):
    print(seq_record.id)
    print(repr(seq_record.seq))
    print(len(seq_record))
'''
from Bio import AlignIO

alignment = AlignIO.read("a_brazil_nut.fasta", "fasta")
print(alignment)


