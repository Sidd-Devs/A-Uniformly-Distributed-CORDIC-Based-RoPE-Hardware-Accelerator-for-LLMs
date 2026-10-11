# ============================================================================
# synth_generic.tcl — one Genus script for every run, driven by environment.
#
# Replaces the eight near-identical per-design scripts. The driver
# (run_all_synth.sh) sets these before invoking genus:
#
#   SYN_DESIGN   top module name
#   SYN_RTL      space-separated RTL files (absolute, or under SYN_RTLDIR)
#   SYN_RTLDIR   directory holding the parameterised RTL copy for this run
#   SYN_SDC      constraint file
#   SYN_LIBBASE  path to gsclib045_all_v4.8
#   SYN_OUT      output directory for this run
#
# Flow, library and effort are identical to the original per-design scripts,
# so results stay comparable with the runs already in hand.
# ============================================================================

set DESIGN   $env(SYN_DESIGN)
set rtlDir   $env(SYN_RTLDIR)
set libBase  $env(SYN_LIBBASE)
set sdcFile  $env(SYN_SDC)
set outRoot  $env(SYN_OUT)

puts "=============================================================="
puts " design : $DESIGN"
puts " rtl    : $env(SYN_RTL)"
puts " sdc    : $sdcFile"
puts " out    : $outRoot"
puts "=============================================================="

set _OUTPUTS_PATH ${outRoot}/outputs
set _REPORTS_PATH ${outRoot}/reports
file mkdir $_OUTPUTS_PATH
file mkdir $_REPORTS_PATH

set_db init_lib_search_path [list \
  ${libBase}/gsclib045/timing \
  ${libBase}/gsclib045_hvt/timing \
  ${libBase}/gsclib045_lvt/timing ]

read_libs [list \
  fast_vdd1v2_basicCells.lib \
  fast_vdd1v2_multibitsDFF.lib \
  fast_vdd1v2_basicCells_hvt.lib \
  fast_vdd1v2_basicCells_lvt.lib ]

read_physical -lefs [list \
  ${libBase}/gsclib045/lef/gsclib045_tech.lef \
  ${libBase}/gsclib045/lef/gsclib045_macro.lef \
  ${libBase}/gsclib045/lef/gsclib045_multibitsDFF.lef \
  ${libBase}/gsclib045_hvt/lef/gsclib045_hvt_macro.lef \
  ${libBase}/gsclib045_lvt/lef/gsclib045_lvt_macro.lef ]

set_db qrc_tech_file ${libBase}/gsclib045_tech/qrc/qx/gpdk045.tch

set_db init_hdl_search_path $rtlDir
set_db max_cpus_per_server 8
set_db information_level 9
set_db tns_opto true

set rtlList {}
foreach f $env(SYN_RTL) { lappend rtlList ${rtlDir}/$f }
read_hdl $rtlList

elaborate  $DESIGN
set_top_module $DESIGN
read_sdc   $sdcFile
check_design -unresolved

syn_generic
syn_map
syn_opt

write_snapshot -innovus -directory $_OUTPUTS_PATH/$DESIGN/opt -tag opt
report_summary -directory        $_REPORTS_PATH/$DESIGN/opt
report_power   >                 $_REPORTS_PATH/$DESIGN/opt/power.rpt
report_area    >                 $_REPORTS_PATH/$DESIGN/opt/area.rpt
report_gates   >                 $_REPORTS_PATH/$DESIGN/opt/gates.rpt

puts "DONE $DESIGN"
exit
