SCENARIOS={
'Normal operation':'Nominal current and ambient conditions without injected faults.',
'Charging-load increase':'Battery current rises after the configured fault onset.',
'Discharging-load increase':'Higher magnitude current models a discharge load step; Joule heating follows I².',
'High ambient temperature':'Ambient is elevated by 12 K for the run.',
'Localized hotspot Zone 1':'Additional heat generation is injected into Zone 1 for a finite interval.',
'Localized hotspot Zone 2':'Additional heat generation is injected into Zone 2 for a finite interval.',
'Cooling actuator degradation':'Zone 1 cooling effectiveness is reduced after fault onset.',
'Temperature sensor bias':'Zone 1 sensor reports a configurable positive bias; plant state remains unchanged.',
'High ambient + hotspot':'Elevated ambient and a localized Zone 1 heat pulse.',
'High load + cooling degradation':'Current increases and Zone 1 cooling effectiveness is reduced.'}
