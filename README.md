# Bone age assessment - Webapp

Deployment ready starter pack for creating fast responsive Web App for Keras bone age assessment models.

Everything packaged in docker with requirement.txt, so you can push it to any docker hosted cloud service.

Also, You can test your changes locally by installing Docker and using the following command:

docker build -t bone-age-assessment . && docker run --rm -it -p 8080:8080 bone-age-assessment

Few dockers hosted services where this starter pack should work =>

* https://render.com
* https://azure.microsoft.com/en-us/services/app-service/containers/
* https://cloud.google.com/cloud-build/docs/
