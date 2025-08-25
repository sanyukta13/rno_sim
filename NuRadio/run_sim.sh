# to run run_sim.py
# remove out_put files from prev runs 
rm data/station_12/output.**
rm -r data/station_12/nu_all_ccnc
rm -r data/station_12/emitter_RNOG_vpol_v3_5inch_center_n1.74/
# run the simulation script
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)
python cleaner_generate_pulser_events.py 12
# python run_sim.py --nur_output --proposal --station_id 11 > run_sim_output.log 2>&1
python simulate_emitter.py --station_id 12 --nur_output > run_sim_output.log 2>&1