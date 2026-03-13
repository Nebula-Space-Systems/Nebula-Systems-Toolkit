# Nebula Tool Kit
# NTK
# Nebulatk

# Goals
* make satellite orbits and propagate them with customizable propagators/force models
* model attitude of a satellite
* calculate spatio-temporo coverage/visibility metrics over areas or the entire world
* model 


# Analysis Studies:

## intersatellite ranges/rates and angles/rates over time
* calculate ranges and range rates between one satellite and another
* calculate doppler shift between one satellite and another

Steps:
* Propagate satellite A and B and get their positions and velocities over time at a small fixed delta time step
* calculate body frame for satellite A over time (LVLH)
* calculate vector from satellite A to satellite B over all scenario times
* project vector from satellite A to satellite B into A's body frame
* take the magnitude of the vector to get range
* finite difference the ranges over each time step to get range rates
* calculate azimuth and elevation from vector from A body to target
* calculate the absolute magnitude angle between azimuth and elevation, and finite difference this angle over time
* using range rates calculate dopler factor from A to B over time


Results/visualization:
* raw timeseries results per pair for ranges/rates, angles/rates, and doppler