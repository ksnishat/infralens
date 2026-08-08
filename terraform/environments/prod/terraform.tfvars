project_name           = "infralens"
environment            = "prod"
location               = "West Europe"
kubernetes_version     = "1.30"
node_count             = 3
vm_size                = "Standard_D4s_v5"
postgres_admin_username = "pgadmin"
postgres_admin_password = "ChangeMeInProd123!"

tags = {
  "Project"     = "InfraLens"
  "Environment" = "prod"
  "Owner"       = "Khaled Saifullah"
  "CostCenter"  = "MLOps-Prod"
}
