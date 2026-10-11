set in_ports  [get_ports -filter {direction == in}]
set out_ports [get_ports -filter {direction == out}]

set_input_delay  0.2 $in_ports
set_output_delay 0.2 $out_ports

set_max_delay 2.0 -from $in_ports -to $out_ports
