#Anything in this file, followed by a period (and an upper-case word), does NOT indicate an end-of-sentence marker.
#Special cases are included for prefixes that ONLY appear before 0-9 numbers.
#Sources: http://nl.wikipedia.org/wiki/Lijst_van_afkortingen 
#         http://nl.wikipedia.org/wiki/Aanspreekvorm
#         http://nl.wikipedia.org/wiki/Titulatuur_in_het_Nederlands_hoger_onderwijs
#any single upper case letter  followed by a period is not a sentence ender (excluding I occasionally, but we leave it in)
#usually upper case letters are initials in a name
A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z

#List of titles. These are often followed by upper-case names, but do not indicate sentence breaks
bacc
bc
bgen
c.i
dhr
dr
dr.h.c
drs
drs
ds
eint
fa
Fa
fam
gen
genm
ing
ir
jhr
jkvr
jr
kand
kol
lgen
lkol
Lt
maj
Mej
mevr
Mme
mr
mr
Mw
o.b.s
plv
prof
ritm
tint
Vz
Z.D
Z.D.H
Z.E
Z.Em
Z.H
Z.K.H
Z.K.M
Z.M
z.v

#misc - odd period-ending items that NEVER indicate breaks (p.m. does NOT fall into this category - it sometimes ends a sentence)
#we seem to have a lot of these in dutch i.e.: i.p.v - in plaats van (in stead of) never ends a sentence
a.g.v
bijv
bijz
bv
d.w.z
e.c
e.g
e.k
ev
i.p.v
i.s.m
i.t.t
i.v.m
m.a.w
m.b.t
m.b.v
m.h.o
m.i
m.i.v
v.w.t

#Numbers only. These should only induce breaks when followed by a numeric sequence
# add NUMERIC_ONLY after the word for this function
#This case is mostly for the english "No." which can either be a sentence of its own, or
#if followed by a number, a non-breaking prefix
Nr #NUMERIC_ONLY# 
Nrs 
nrs
nr #NUMERIC_ONLY#
