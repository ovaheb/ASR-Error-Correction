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

cv_examples = ["""Example 1:\n
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

lrs_examples = ["""Example 1:\n
<hypothesis1> even more beautiful </hypothesis1>\n
<hypothesis2> to even more beautiful people </hypothesis2>\n
<hypothesis3> even more beautiful beautiful </hypothesis3>\n
<hypothesis4> even more beautiful people </hypothesis4>\n
<hypothesis5> even more beautiful beauty </hypothesis5>\n
\nYour output: even more beautiful \n\n""",
"""Example 2:\n
<hypothesis1> before the days of gcf </hypothesis1>\n
<hypothesis2> than before the days of gcf </hypothesis2>\n
<hypothesis3> for the days of gcf </hypothesis3>\n
<hypothesis4> before the days of gcf </hypothesis4>\n
<hypothesis5> the days of gcf </hypothesis5>\n
\nYour output: before the days of gcse \n\n""",
"""Example 3:\n
<hypothesis1> in the eighteenth century </hypothesis1>\n
<hypothesis2> in the eighteenth century this </hypothesis2>\n
<hypothesis3> in the eighteenth century did not it </hypothesis3>\n
<hypothesis4> in the eighteenth century it is </hypothesis4>\n
<hypothesis5> in the eighteenth century it is </hypothesis5>\n
\nYour output: in the eighteenth century \n\n""",
"""Example 4:\n
<hypothesis1> and suspicion of concealing nine hundred and seventy-five thousand </hypothesis1>\n
<hypothesis2> and suspicion of conceding nine hundred and seventy-five thousand </hypothesis2>\n
<hypothesis3> and suspicion of concealing </hypothesis3>\n
<hypothesis4> and suspicion of conceding nine hundred and seventy-five thousand </hypothesis4>\n
<hypothesis5> and suspicion of concealing nine hundred and seventy-five thousand </hypothesis5>\n
\nYour output: on suspicion of concealing nine hundred and seventy-five \n\n""",
"""Example 5:\n
<hypothesis1> we love the collectors </hypothesis1>\n
<hypothesis2> we love the collectors we </hypothesis2>\n
<hypothesis3> we love to collect as we </hypothesis3>\n
<hypothesis4> we love the collectors </hypothesis4>\n
<hypothesis5> we love to collect as we </hypothesis5>\n
\nYour output: we love the collectors \n\n""",
"""Example 6:\n
<hypothesis1> i want to be wowed by something </hypothesis1>\n
<hypothesis2> i want to be wired by something </hypothesis2>\n
<hypothesis3> i want to be wild by something </hypothesis3>\n
<hypothesis4> i want to be wowed by something </hypothesis4>\n
<hypothesis5> i want to be </hypothesis5>\n
\nYour output: i want to be wowed by something \n\n""",
"""Example 7:\n
<hypothesis1> and a little bit of what the fnc does you are good </hypothesis1>\n
<hypothesis2> a little bit of what you have to see those you are good </hypothesis2>\n
<hypothesis3> a little bit of what the fnc does you are good </hypothesis3>\n
<hypothesis4> a little bit of what you have to see those you are good at </hypothesis4>\n
<hypothesis5> and a little bit of what you have to see those you are good at </hypothesis5>\n
\nYour output: a little bit of what you fancy does you good \n\n"""
]
chime_examples = ["""Example 1:\n
<hypothesis1> most of those exceeding have stuck it out through deficits </hypothesis1>\n
<hypothesis2> most of those exceeding have stuck it out through decades </hypothesis2>\n
<hypothesis3> most of those exceeding have stuck it out through breakfasts </hypothesis3>\n
<hypothesis4> most of those exceeding have stuck it out through dead cells </hypothesis4>\n
<hypothesis5> most of those succeeding have stuck it out through their sales </hypothesis5>\n
\nYour output: most of those succeeding have stuck it out through deficits \n\n""",
"""Example 2:\n
<hypothesis1> all of these waiting experience failed at this event </hypothesis1>\n
<hypothesis2> all of these retail experience failed at this event </hypothesis2>\n
<hypothesis3> all of these retailing experience failed at this event </hypothesis3>\n
<hypothesis4> all of these retainer experience failed at this event </hypothesis4>\n
<hypothesis5> all of these retail experience failed at this event </hypothesis5>\n
\nYour output: all of these retailing experiments failed miserably \n\n""",
"""Example 3:\n
<hypothesis1> first aid officials could not be reached </hypothesis1>\n
<hypothesis2> first day officials could not be reached </hypothesis2>\n
<hypothesis3> first state officials could not be reached </hypothesis3>\n
<hypothesis4> first aid officials could not be reached </hypothesis4>\n
<hypothesis5> first state officials could not be reached </hypothesis5>\n
\nYour output: first state officials could not be reached \n\n""",
"""Example 4:\n
<hypothesis1> i am about to become george bush for a night </hypothesis1>\n
<hypothesis2> i am about to become george bush for a name </hypothesis2>\n
<hypothesis3> i am about to become george bush for a night </hypothesis3>\n
<hypothesis4> i am about to become george bush for a night </hypothesis4>\n
<hypothesis5> i am about to become george bush for a name </hypothesis5>\n
\nYour output: i am about to become george bush for a night \n\n""",
"""Example 5:\n
<hypothesis1> i have been trying to live my life by an idea she says hoping to a video game </hypothesis1>\n
<hypothesis2> i have been trying to live my life by an idea she says pointing to a video game </hypothesis2>\n
<hypothesis3> i have been trying to live my life by an idea she says poking to a video game </hypothesis3>\n
<hypothesis4> i have been trying to live my life by an idea she says pointing to a video game </hypothesis4>\n
<hypothesis5> i have been trying to live my life by an idea she says poaking to a video game </hypothesis5>\n
\nYour output: i have been trying to live my life by an idea she says pointing to a video 
game \n\n""",
"""Example 6:\n
<hypothesis1> the frac a new york based development company run by samuel j the frac and declan declan </hypothesis1>\n
<hypothesis2> the frac a new york based development company run by samuel j lefrac and deconan deconan </hypothesis2>\n
<hypothesis3> the frac a new york based development company run by samuel jay the frac and decony decony </hypothesis3>\n
<hypothesis4> the frac a new york based development company run by samuel j the frac and declan declan </hypothesis4>\n
<hypothesis5> the frac a new york based development company run by samuel jay the frac and declan declan </hypothesis5>\n
\nYour output: lefrak a new york based development company run by samuel j lefrak declined to comment \n\n""",
"""Example 7:\n
<hypothesis1> all of these waiting experience failed at this event </hypothesis1>\n
<hypothesis2> all of these retail experience failed at this event </hypothesis2>\n
<hypothesis3> all of these retailing experience failed at this event </hypothesis3>\n
<hypothesis4> all of these retainer experience failed at this event </hypothesis4>\n
<hypothesis5> all of these retail experience failed at this event </hypothesis5>\n
\nYour output: all of these retailing experiments failed miserably \n\n"""]