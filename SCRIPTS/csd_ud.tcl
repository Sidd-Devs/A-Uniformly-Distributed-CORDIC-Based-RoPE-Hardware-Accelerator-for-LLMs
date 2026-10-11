
if {[file exists /proc/cpuinfo]} {
  sh grep "model name" /proc/cpuinfo
  sh grep "cpu MHz"    /proc/cpuinfo
}

puts "Hostname : [info hostname]"

set DESIGN rope_csd_pair
set SYN_EFF medium 
set MAP_EFF medium
set OPT_EFF medium

set RELEASE [lindex [get_db program_version] end]
set _OUTPUTS_PATH OUTPUT/outputs_${RELEASE}
set _REPORTS_PATH OUTPUT/reports_${RELEASE}

if {![file exists ${_OUTPUTS_PATH}]} {
  file mkdir ${_OUTPUTS_PATH}
  puts "Creating directory ${_OUTPUTS_PATH}"
}

if {![file exists ${_REPORTS_PATH}]} {
  file mkdir ${_REPORTS_PATH}
  puts "Creating directory ${_REPORTS_PATH}"
}


set rtlDir /home/redhatacademy19/Documents/Siddhant/RTL

set scriptDir /home/redhatacademy19/Documents/Siddhant/SCRIPTS

set libDir [list \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045/timing    \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045_hvt/timing \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045_lvt/timing  \  
  ]

set libList [list \
         fast_vdd1v2_basicCells.lib \
         fast_vdd1v2_multibitsDFF.lib \
         fast_vdd1v2_basicCells_hvt.lib \
         fast_vdd1v2_basicCells_lvt.lib  ]  


set_db init_lib_search_path $libDir
set_db script_search_path { $scriptDir } 
set_db init_hdl_search_path {$rtlDir} 

set_db max_cpus_per_server 8 

#set_db syn_generic_effort $SYN_EFF 
#set_db syn_map_effort $MAP_EFF 
#set_db syn_opt_effort $OPT_EFF 

set_db information_level 9 
# set_db pbs_mmmc_flow true

set_db tns_opto true 
#set_db lp_insert_clock_gating true



puts "Now load RTL LIST"

set rtlList [list \
  ${rtlDir}/csd_ud_stage.v \
  ${rtlDir}/csd_ud_cordic_core.v \
  ${rtlDir}/rope_csd_pair.v \
  ${rtlDir}/csd_control.v \
]


 

## -ing in MMMC defination file and lef files
# read_mmmc $scriptDir/mmmc.tcl
read_libs $libList

read_physical -lefs { \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045/lef/gsclib045_tech.lef \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045/lef/gsclib045_macro.lef \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045/lef/gsclib045_multibitsDFF.lef \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045_hvt/lef/gsclib045_hvt_macro.lef \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045_lvt/lef/gsclib045_lvt_macro.lef
}

set_db qrc_tech_file /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045_tech/qrc/qx/gpdk045.tch

# Reading hdl files, initialize the database and elaborating them
read_hdl  $rtlList

elaborate $DESIGN

set_top_module $DESIGN
#read_def ../DEF/dtmf.def
read_sdc /home/redhatacademy19/Documents/Siddhant/CONSTRAINTS/rope.sdc

#init_design
#time_info init_design
check_design -unresolved

## Set the innovus executable to be used for placement and routing
## set_db innovus_executable  <Innovus Executables>


####################################################################################################
## Synthesizing the design
####################################################################################################

syn_generic

write_snapshot -directory $_OUTPUTS_PATH/$DESIGN/generic -tag generic
report_summary -directory $_REPORTS_PATH/$DESIGN/generic
report_power > $_REPORTS_PATH/$DESIGN/generic/power.rpt
puts "Runtime & Memory after 'syn_generic'"
time_info GENERIC

syn_map

write_snapshot -directory $_OUTPUTS_PATH/$DESIGN/mapped -tag mapped
report_summary -directory $_REPORTS_PATH/$DESIGN/mapped
report_power > $_REPORTS_PATH/$DESIGN/mapped/power.rpt
puts "Runtime & Memory after 'syn_map'"
time_info MAPPED

syn_opt

## generate reports to save the Innovus stats
write_snapshot -innovus -directory $_OUTPUTS_PATH/$DESIGN/opt -tag opt
report_summary -directory $_REPORTS_PATH/$DESIGN/opt
report_power > $_REPORTS_PATH/$DESIGN/opt/power.rpt
puts "Runtime & Memory after syn_opt"
time_info OPT

## write out the final database
write_db -to_file ${DESIGN}.db

puts "Final Runtime & Memory."
time_info FINAL
puts "============================"
puts "Synthesis Finished ........."
puts "============================"

#quit
