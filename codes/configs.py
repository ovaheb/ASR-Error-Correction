cv_generation_config = {"max_tokens": 25, "temperature": 0.9}
wsj_generation_config = {"max_tokens": 30, "temperature": 0.9}
swbd_generation_config = {"max_tokens": 65, "temperature": 0.9}
atis_generation_config = {"max_tokens": 45, "temperature": 0.9}
td_generation_config = {"max_tokens": 130, "temperature": 0.9}
ls_clean_generation_config = {"max_tokens": 100, "temperature": 0.9}
ls_others_generation_config = {"max_tokens": 130, "temperature": 0.9}
lrs_generation_config = {"max_tokens": 25, "temperature": 0.9}
chime_generation_config = {"max_tokens": 30, "temperature": 0.9}

small_generation_config = {"max_tokens": 20, "temperature": 0.9}
moderate_generation_config = {"max_tokens": 200, "temperature": 0.9}
deepseek_generation_config = {"max_tokens": 2500, "temperature": 0.9}

cv_examples = ["""
Example 1:\n
<hypothesis1> see stongers were executed for these crimes and manures devoted to other islands </hypothesis1>\n
<hypothesis2> the stungers were executed for this crime and maneuvers devoted to other islands </hypothesis2>\n
<hypothesis3> the stungers were executed for this crime and many were deported to other islands </hypothesis3>\n
<hypothesis4> the strongest were executed for the crime and maneuvers deported to other islands </hypothesis4>\n
<hypothesis5> the stungers were executed for these crimes and maneuvers devoted to other islands </hypothesis5>\n
\nYour output: six tongans were executed for this crime and many were deported to other islands\n\n""",
"""Example 2:\n
<hypothesis1> the hamlet of whitewell likes to the west </hypothesis1>\n
<hypothesis2> the hamlet of white will lights to the west </hypothesis2>\n
<hypothesis3> the hamlet of whitewell lies to the west</hypothesis3>\n
<hypothesis4> the hamlet of whitewill lies to the west </hypothesis4>\n
<hypothesis5> the hamlet of whiteville likes to the west </hypothesis5>\n
\nYour output: the hamlet of whitewell lies to the west\n\n""",
"""Example 3:\n
<hypothesis1> conway was farmed and disguised as conway </hypothesis1>\n
<hypothesis2> konui was formed and disguised as konui </hypothesis2>\n
<hypothesis3> conroy was formed and disguised as conway </hypothesis3>\n
<hypothesis4> conway was formed and disguised as conway </hypothesis4>\n
<hypothesis5> connolly was formed and disguised as conway </hypothesis5>\n
\nYour output: conwy was formerly anglicized as conway \n\n""",
"""Example 4:\n
<hypothesis1> due to space limitations as an extremely narrow platform </hypothesis1>\n
<hypothesis2> due to space limitation as an extremely narrow platform </hypothesis2>\n
<hypothesis3> due to space limitation as an extremely narrow platform </hypothesis3>\n
<hypothesis4> due to space limitations as an extremely narrow platform </hypothesis4>\n
<hypothesis5> due to space limitations as an extremely narrow platform </hypothesis5>\n
\nYour output: due to space limitations it has an extremely narrow platform \n\n""",
"""Example 5:\n
<hypothesis1> the band continues to tune nationally </hypothesis1>\n
<hypothesis2> the band continues to tour nationally </hypothesis2>\n
<hypothesis3> the band continues to do it nationally </hypothesis3>\n
<hypothesis4> the band continues to tour nationally </hypothesis4>\n
<hypothesis5> the band continues to do it nationally </hypothesis5>\n
\nYour output: the band continues to tour nationally \n\n""",
"""Example 6:\n
<hypothesis1> around holdcroft requires a lot more skill to keep upright </hypothesis1>\n
<hypothesis2> around holdcroft requires a lot more skill to keep upright </hypothesis2>\n
<hypothesis3> around hold croft requires a lot more skill to keep upright </hypothesis3>\n
<hypothesis4> a roundhold craft requires a lot more skill to keep upright </hypothesis4>\n
<hypothesis5> around hold craft requires a lot more skill to keep upright </hypothesis5>\n
\nYour output: a round hulled craft requires a lot more skill to keep upright \n\n""",
"""Example 7:\n
<hypothesis1> tom the montana is a collective term for the appland varieties e g </hypothesis1>\n
<hypothesis2> tom the monten is a collective term for the upland varieties e g </hypothesis2>\n
<hypothesis3> tom the monten is a collective term for the appland varieties e g </hypothesis3>\n
<hypothesis4> tom the monten is a collective term for the appland varieties e g </hypothesis4>\n
<hypothesis5> tom the montana is a collective term for the appland varieties e g </hypothesis5>\n
\nYour output: tomme de montagne is a collective term for the upland varieties e g \n\n"""]

lrs_examples = []
chime_examples = []