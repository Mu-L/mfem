make all -j 8
./J_field_projector_J_tor_direct&
./J_field_projector_J_pol_direct&
./B_field_projector_B_tor_CG && ./J_field_projector_J_pol_Hdiv&
./B_field_projector_B_tor_DG && ./J_field_projector_J_pol_Hcurl&
./B_field_projector_B_pol_Hdiv && ./J_field_projector_J_tor_Hdiv&
./B_field_projector_B_pol_Hcurl && ./J_field_projector_J_tor_Hcurl
./JxB_validator_JxB_tor_A & ./JxB_validator_JxB_pol_A &
./JxB_validator_JxB_tor_B & ./JxB_validator_JxB_pol_B
./JxB_validator_grad_p