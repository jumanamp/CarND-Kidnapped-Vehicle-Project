/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// 1- Set number of particles
	num_particles = 75;

	// 2- Define sensor noise normal distributions, with mean as GPS value
  normal_distribution<double> x_N_dist(x, std[0]);
  normal_distribution<double> y_N_dist(y, std[1]);
  normal_distribution<double> theta_N_dist(theta, std[2]);

	// 3- Initialize particle filter
	for (int i = 0; i < num_particles; i++) {
		// Create a particle
		Particle p;
		p.id = i;
		// Add noise to the values
		p.x = x_N_dist(gen);
		p.y = y_N_dist(gen);
		p.theta = theta_N_dist(gen);
		p.weight = 1.0;

		// Update particles vector
		particles.push_back(p);
		weights.push_back(p.weight);
	}
	// 4- Initialization done
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	//  Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // 1- Use motion model to calculate new state of the particle
	for (int i = 0; i < num_particles; i++) {
		// Get particle state
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// calculate new state
		double pred_x, pred_y, pred_theta;
    if (fabs(yaw_rate) < 0.00001) {
      pred_x = p_x + velocity * delta_t * cos(p_theta);
      pred_y = p_y + velocity * delta_t * sin(p_theta);
			pred_theta = p_theta;
    }
    else {
      pred_x = p_x + (velocity / yaw_rate) * (sin(p_theta + yaw_rate * delta_t) - sin(p_theta));
      pred_y = p_y + (velocity / yaw_rate) * (cos(p_theta) - cos(p_theta + yaw_rate * delta_t));
      pred_theta = p_theta + (yaw_rate * delta_t);
    }

    // 2- Define prediction noise normal distributions, with mean as predicted values
		normal_distribution<double> x_N_dist(pred_x, std_pos[0]);
	  normal_distribution<double> y_N_dist(pred_y, std_pos[1]);
	  normal_distribution<double> theta_N_dist(pred_theta, std_pos[2]);

    // 3- add noise
    particles[i].x = x_N_dist(gen);
    particles[i].y = y_N_dist(gen);
    particles[i].theta = theta_N_dist(gen);

	}

}

void ParticleFilter::dataAssociation(double sensor_range, std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {

	    // 1- Get an observation landmark
	    LandmarkObs obs_lm = observations[i];

	    // Initialize distance to maximum possible value for double value
			// This will determine the minimum most distanced prediction point.
	    double pred_dist_min = sensor_range * 2;
	    // id of the minimum most distanced prediction point
	    int pred_map_id = -1;

	    for (int j = 0; j < predicted.size(); j++) {
	      // Get a predicted landmark
	      LandmarkObs pred_lm = predicted[j];
	      // Compute distance between the predicted landmark to observed landmark in hand
	      double lm_dist = dist(obs_lm.x, obs_lm.y, pred_lm.x, pred_lm.y);

	      // iterate to find the predicted landmark nearest to the observed landmark in hand
	      if (lm_dist < pred_dist_min) {
	        pred_dist_min = lm_dist;
	        pred_map_id = pred_lm.id;
	      }
	    }
	    // set the observation's id to the nearest predicted landmark's id
	    observations[i].id = pred_map_id;
	  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // For resampling, we will normalize weights to be within [0,1]
  double w_normalizer = 0.0;

	for (int i = 0; i < num_particles; i++) {

		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

    // Transform the observations from vehicle coordinates to map coordinates
		vector<LandmarkObs> obs_transformed;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double x_o_map = (cos(p_theta)*observations[j].x) - (sin(p_theta)*observations[j].y) + p_x;
      double y_o_map = (sin(p_theta)*observations[j].x) + (cos(p_theta)*observations[j].y) + p_y;
      obs_transformed.push_back(LandmarkObs{ j, x_o_map, y_o_map });
    }

	  // Only consider the map landmark locations within sensor range of the particle
    vector<LandmarkObs> map_predictions;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      float x_map_lm = map_landmarks.landmark_list[j].x_f;
      float y_map_lm = map_landmarks.landmark_list[j].y_f;
      int id_map_lm = map_landmarks.landmark_list[j].id_i;

			// choose the observations within sensor range, check x and y distance instead of dist fucntion
			// which is computationally expensive
			if (fabs(x_map_lm - p_x) <= sensor_range && fabs(y_map_lm - p_y) <= sensor_range) {
        map_predictions.push_back(LandmarkObs{ id_map_lm, x_map_lm, y_map_lm });
      }
    }

		// Call dataAssociation for the predicted and transformed landmark observations
		// on current particle
		dataAssociation(sensor_range, map_predictions, obs_transformed);

		// Update the weight of the particle.
    particles[i].weight = 1.0;
		double x_std = std_landmark[0];
		double y_std = std_landmark[1];
		double norm = ( 1/(2 * M_PI * x_std * y_std));
		for (unsigned int j = 0; j < obs_transformed.size(); j++) {
			// Get observation landmark info
			double x_obs_lm = obs_transformed[j].x;
			double y_obs_lm = obs_transformed[j].y;
			double id_lm = obs_transformed[j].id;

			// Get the prediction landmark info that matches the id of the observation.
      double x_pred_lm, y_pred_lm;
			for(unsigned int k = 0; k < map_predictions.size(); k++) {
				if(map_predictions[k].id == id_lm){
					x_pred_lm = map_predictions[k].x;
					y_pred_lm = map_predictions[k].y;

				 // Compute Weight for the observation landmark using the multivariate Guassian Approach
					double x_diff = x_pred_lm -  x_obs_lm;
					double y_diff = y_pred_lm -  y_obs_lm;
		      double w =  norm * exp( -1.0 * ( pow(x_diff, 2) /(2 * pow(x_std, 2)) + ( pow(y_diff, 2)/ (2 * pow(y_std, 2) ) ) ) );

		      // Update weight
		      particles[i].weight *= w;
				}
			}
		}
		// Update normalizer to keep summing the weights
    w_normalizer += particles[i].weight;
	}
	// Normalize all particle weights
	for (int i = 0; i<particles.size(); i++) {
		particles[i].weight /= w_normalizer;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Particle Survival based on Weights
  vector<Particle> particles_updated;

  // Generate an index for starting the resampling wheel.
	std::uniform_int_distribution<int> uni_int_dist(0, num_particles-1);
  int index = uni_int_dist(gen);

	// Create a discrete disrtibution of weights.
	std::uniform_real_distribution<double> uni_real_dist(0.0, *max_element(weights.begin(), weights.end()));

  // Resample wheel from lecture code
	double beta = 0.0;
	for(int i = 0; i < particles.size(); i++) {
		beta += uni_real_dist(gen) * 2.0;
		while(beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		particles_updated.push_back(particles[index]);
	}
	// Update particles with new particles.
	particles = particles_updated;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
