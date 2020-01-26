/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::discrete_distribution;
using std::numeric_limits;
using std::uniform_int_distribution;
using std::uniform_real_distribution;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 1000;  // Set the number of particles

  // Generate random engine
  default_random_engine gen;

  // Gaussian distributions for x, y and theta for particle noise
  normal_distribution<float> dist_x(x, std[0]);
  normal_distribution<float> dist_y(y, std[1]);
  normal_distribution<float> dist_theta(theta, std[2]);

  // Initialize particles
  for (unsigned int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = x + dist_x(gen);
    p.y = y + dist_y(gen);
    p.theta = theta + dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
    //cout << p.id << " | " << p.x << " | " << p.y << " | " << p.theta << " | " << p.weight << endl;
  }

  // Set initialized variable to true
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

   // Random engine
   default_random_engine gen;

   // Gaussian distribution for noise of measurement
   normal_distribution<float> dist_x(0, std_pos[0]);
   normal_distribution<float> dist_y(0, std_pos[1]);
   normal_distribution<float> dist_theta(0, std_pos[2]);

   // Loop through particles
   for(unsigned int i = 0; i < num_particles; ++i) {

     if (fabs(yaw_rate) < 0.00001) {
       // Updating x, y and yaw angle, when yaw rate is equal to zero
       particles[i].x += velocity * delta_t * cos(particles[i].theta);
       particles[i].y += velocity * delta_t * sin(particles[i].theta);

     } else {
       // Updating x, y and yaw angle, when yaw rate is not equal to zero
       particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
       particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
       particles[i].theta += yaw_rate * delta_t;

     }

     // Add noise to the particles
     particles[i].x += dist_x(gen);
     particles[i].y += dist_y(gen);
     particles[i].theta += dist_theta(gen);
     //cout << particles[i].id << " | " << particles[i].x << " | " << particles[i].y << " | " << particles[i].theta << " | " << particles[i].weight << endl;
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

   // Loop through observations
	 for(unsigned int i = 0; i < observations.size(); ++i) {
  		float min_dist = numeric_limits<float>::max();
  		int closest_id = -1;
  		float obs_x = observations[i].x;
  		float obs_y = observations[i].y;

      // Loop through predictions
		  for (unsigned int j = 0; j < predicted.size(); ++j) {
    		  float pred_x = predicted[j].x;
    		  float pred_y = predicted[j].y;
    		  int pred_id = predicted[j].id;
    		  float current_dist = dist(obs_x, obs_y, pred_x, pred_y);

    		  if (current_dist < min_dist) {
    		      min_dist = current_dist;
    		      closest_id = pred_id;
  		    }
		  }
		  observations[i].id = closest_id;
	 }
}

double multi_gaussian(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) {
  /*
   * Multivariate Gaussian Distribution
   */
  return exp( -((x - mu_x) * (x - mu_x) / (2 * sig_x * sig_x) + (y - mu_y) * (y - mu_y) / (2 * sig_y * sig_y))) / (2.0 * M_PI * sig_x * sig_y);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

   // Empty weights vector
   weights.clear();

   // Loop through all particles
   for(unsigned int i = 0; i < particles.size(); ++i) {

     //
     // 1) Transform observations from vehicle coordinates to map coordinates
     //

     vector<LandmarkObs> transformed_observations;

     for(unsigned int j = 0; j < observations.size(); ++j) {
         LandmarkObs temp;

         temp.x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
         temp.y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
         temp.id = -1;

         transformed_observations.push_back(temp);
     }

     //
     // 2) Filter map landmarks
     //
     vector<LandmarkObs> predicted_landmarks;

     for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
         float landmark_dist;

         landmark_dist = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);

         if(landmark_dist < sensor_range) {
             LandmarkObs pred_landmark;
             pred_landmark.id = map_landmarks.landmark_list[j].id_i;
             pred_landmark.x = map_landmarks.landmark_list[j].x_f;
             pred_landmark.y = map_landmarks.landmark_list[j].y_f;
             predicted_landmarks.push_back(pred_landmark);
         }
     }

     //
     // 3) Associate observations with nearest Neighbor algorithm
     //
     dataAssociation(predicted_landmarks, transformed_observations);

     //
     // 4) Calculate the weight of each particle with Multivariate Gaussian probability
     //
     double probability = 1;
     double multi_gauss_prob;

     for(unsigned int j = 0; j < predicted_landmarks.size(); ++j) {
       int closest_id = -1;
       double min_dist = numeric_limits<double>::max();

       double px = predicted_landmarks[j].x;
       double py = predicted_landmarks[j].y;

       for(unsigned int k = 0; k < transformed_observations.size(); ++k) {
         double tx = transformed_observations[k].x;
         double ty = transformed_observations[k].y;
         double curr_dist = dist(px, py, tx, ty);

         if(curr_dist < min_dist) {
             min_dist = curr_dist;
             closest_id = k;
         }
       }

       if (closest_id != -1){
         multi_gauss_prob = multi_gaussian(px, py, transformed_observations[closest_id].x, transformed_observations[closest_id].y, std_landmark[0], std_landmark[1]);
         probability *= multi_gauss_prob;
       }
     }
     //
     // 5) Update weight
     //
     weights.push_back(probability);
     particles[i].weight = probability;

   }

}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

   // Vector for resampled particles
   vector<Particle> resampled_particles;

 	// Random engine
 	default_random_engine gen;

 	// Get random index for particle
 	uniform_int_distribution<int> particle_index(0, num_particles - 1);
 	int current_index = particle_index(gen);

 	double beta = 0.0;
 	double max_weight_2 = 2.0 * *max_element(weights.begin(), weights.end());

  // Pick out of bag of particles with replacement (See lecture)
 	for(unsigned int i = 0; i < particles.size(); ++i) {
   		uniform_real_distribution<double> random_weight(0.0, max_weight_2);
   		beta += random_weight(gen);

   	  while (beta > weights[current_index]) {
     	    beta -= weights[current_index];
     	    current_index = (current_index + 1) % num_particles;
   	  }
   	  resampled_particles.push_back(particles[current_index]);
 	}
 	particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
      v = best.sense_x;
    } else {
      v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
