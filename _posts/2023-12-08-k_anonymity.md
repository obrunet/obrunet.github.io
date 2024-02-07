---
title: "Practical examples on why k-anonymity are not enough"
date: 2023-12-08
categories:
  - Privacy
tags: [Privacy]
header:
  image: "./img/pexels-scott-webb-430208.jpg"
excerpt: "Even improved version of K-anonymity with L-diversity & T-closeness can be attacked..."
mathjax: "true"
---
/images/2023-12-08-k_anonymity/
Banner created from a photo of [Scott Webb on Pexels.com](https://www.pexels.com/fr-fr/photo/deux-cameras-de-securite-grey-bullet-430208/)


![image info](./img/pexels-scott-webb-430208.jpg)

why sharing data?

quasi indentifiers
• What	is	a	quasi-idenWfier?	
– CombinaWon	of	adributes	(that	an	adversary	may	know)	that	uniquely	
idenWfy	a	large	fracWon	of	the	populaWon.	
– There	can	be	many	sets	of	quasi-idenWfiers.			
If	Q	=	{B,	Z,	S}	is	a	quasi-idenWfier,	then	Q	+	{N}	is	also	a	quasi-idenWfier.



We	saw	examples	before	…	

- Massachuseds	governor	adack	
• AOL	privacy	breach	
• Neglix	adack	
• Social	Network	adacks


# K-anonymity

Each released record should be indistinguishable
from at least (k-1) others on its QI attributes
• Alternatively: cardinality of any query result on
released data should be at least k
• k-anonymity is (the first) one of many privacy
definitions in this line of work
– l-diversity, t-closeness, m-invariance, delta-presence...



![image info](./img/1.jpg


coarsening
![image info](./img/2.jpg

clusering
![image info](./img/3.jpg


micro agg
![image info](./img/4.jpg


Joining	the	published	data	to	an	external	dataset	using	quasiidenWfiers	results	in	at	least	k	records	per	quasi-idenWfier	
combinaWon.	
– Need	to	guarantee	k-anonymity	against	the	largest	set	of	quasi-idenWfiers


## Does	k-Anonymity	guarantee sufficient	privacy	?

homo attck

![image info](./img/5.jpg


background

![image info](./img/6.jpg
![image info](./img/7.jpg
![image info](./img/8.jpg




# L-Diversity:	Privacy	Beyond	K-Anonymity	

L-diversity principle: A q-block is l-diverse if
contains at least l ‘well represented” values for
the sensitive attribute S. A table is l-diverse if
every q-block is l-diverse


L-Diversity	Principle:		
Every	group	of	tuples	with	the	same	Q-ID	values	has		
≥	L	disInct	“well	represented”	sensiIve	values.		
Ques%ons:	
• What	kind	of	adversarial	adacks	do	we	guard	against?	
• Why	is	this	the	right	definiWon	for	privacy?	
– What	does	the	parameter	L	signify?	

![image info](./img/9.jpg

Privacy	SpecificaWon	for	L-Diversity	
• The	link	between	idenWty	and	adribute	value	is	the	sensiWve	
informaWon.		
							 			“Does	Bob	have	Cancer?	Heart	disease?	Flu?”	
									“Does	Umeko	have	Cancer?	Heart	disease?	Flu?”	
• Adversary	knows	≤	L-2	negaWon	statements.	
				“Umeko	does	not	have	Heart	Disease.”	
– Data	Publisher	may	not	know	exact	adversarial	knowledge	
• Privacy	is	breached	when	idenWty	can	be	linked	to	adribute	value	
with	high	probability	
				Pr[	“Bob	has	Cancer”	|	published	table,	adv.	knowledge]	>	t



Therefore,	in	order	for	privacy,			
check	for	each	individual	u,	and	each	disease	s	
	Pr[	“u	has	disease	s”	|	T*,		adv.	knowledge	about	u]			<		t	
And	we	are	done	…	??	
25	
Data	publisher	does	not	know	the		adversary’s	
knowledge	about	u	
• Different	adversaries	have	varying	amounts	of	knowledge.	
•	Adversary	may	have	different	knowledge	about	different	
individuals.	
adv.


Distinct l-diversity
– Each equivalence class has at least l well-represented sensitive
values
– Limitation:
• Doesn’t prevent the probabilistic inference attacks
• Ex.
In one equivalent class, there are ten tuples. In the “Disease” area,
one of them is “Cancer”, one is “Heart Disease” and the remaining
eight are “Flu”. This satisfies 3-diversity, but the attacker can still
affirm that the target person’s disease is “Flu” with the accuracy of
80%.

Entropy l-diversity
– Each equivalence class not only must have enough different
sensitive values, but also the different sensitive values must
be distributed evenly enough.
– It means the entropy of the distribution of sensitive values in
each equivalence class is at least log(l)
– Sometimes this maybe too restrictive. When some values
are very common, the entropy of the entire table may be
very low. This leads to the less conservative notion of ldiversity.

Recursive (c,l)-diversity
– The most frequent value does not appear too frequently
– r1 <c(rl +rl+1 +…+rm )

### Limitations
l-diversity may be difficult and unnecessary to achieve.
 A single sensitive attribute
 Two values: HIV positive (1%) and HIV negative
(99%)
 Very different degrees of sensitivity
 l-diversity is unnecessary to achieve
 2-diversity is unnecessary for an equivalence class
that contains only negative records
 l-diversity is difficult to achieve
 Suppose there are 10000 records in total
 To have distinct 2-diversity, there can be at most
10000*1%=100 equivalence classes

# L-Diversity:
Guarding	against	unknown	adversarial	knowledge.	
Limit	adversarial	knowledge	
– Knows	≤	(L-2)	negaWon	statements	of	the	form	
“Umeko	does	not	have	a	Heart	disease.”	
• Consider	the	worst	case	
– Consider	all	possible	conjuncWons	of		≤	(L-2)	statements	


# T-Closeness

k-anonymity prevents identity disclosure but not
attribute disclosure
• To solve that problem l-diversity requires that each
eq. class has at least l values for each sensitive
attribute
• But l-diversity has some limitations
• t-closeness requires that the distribution of a
sensitive attribute in any eq. class is close to the
distribution of a sensitive attribute in the overall table.

Privacy is measured by the information gain of an
observer.
– Information Gain = Posterior Belief – Prior Belief
– Q = the distribution of the sensitive attribute in the whole
table
– P = the distribution of the sensitive attribute in eq. class

Principle:
– An equivalence class is said to have t-closeness
• if the distance between the distribution of a sensitive
attribute in this class and the distribution of the attribute
in the whole table is no more than a threshold t.
– A table is said to have t-closeness
• if all equivalence classes have t-closeness.

Conclusion
• t-closeness protects against attribute
disclosure but not identity disclosure
• t-closeness requires that the distribution of a
sensitive attribute in any eq. class is close to
the distribution of a sensitive attribute in the
overall table.

![image info](./img/4.jpg

![image info](./img/4.jpg

![image info](./img/4.jpg

![image info](./img/4.jpg

![image info](./img/4.jpg

# References:
- [Other Privacy Definitions: l-diversity and t-closeness by Murat Kantarcioglu](https://personal.utdallas.edu/~muratk/courses/dbsec09s_files/DBSec_priv3.pdf)
- [Measures	of Anonymity/Privacy: k-Anonymity, L-Diversity, t-Closeness by Ashwin Machanavajjhala (Duke University)](https://courses.cs.duke.edu/fall13/compsci590.3/slides/lec4.pdf)
- Programming Differential Privacy - [Website](https://programming-dp.com/cover.html) & [Github repo](https://github.com/uvm-plaid/programming-dp)
- [K-anonymity, the parent of all privacy definitions by Damien Desfontaines](https://desfontain.es/privacy/k-anonymity.html)
