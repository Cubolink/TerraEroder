//
// Created by major on 28-12-2021.
//

#ifndef TERRAINGEN_GEOMETRY_H
#define TERRAINGEN_GEOMETRY_H

#include <iostream>

namespace Geometry
{


    template<class T>
    class Vector
    {
    private:
        T x, y, z;
    public:
        Vector(T x, T y, T z)
                : x(x), y(y), z(z)
        {
        }

        T getX() const;

        T getY() const;

        T getZ() const;

        Vector<T> productoEscalar(const T &s) const;

        T productoPunto(const Vector<T> &v) const;

        Vector<T> productoCruz(const Vector<T> &v) const;

        bool operator==(const Vector<T> &v) const;

        bool operator!=(const Vector<T> &v) const;

        Vector<T>& operator=(const Vector<T> &v);

        Vector<T> operator+(const Vector<T> &v) const;

        Vector<T> operator-() const;

        Vector<T> operator-(const Vector<T> &v) const;

        Vector<T> operator*(const T &s) const;

        T operator*(const Vector<T> &v) const;

        T moduloCuadrado() const;

        double modulo() const;

        Vector<double> normalizado();

        ~Vector() = default;

    };


// Definitions


    template<class T>
    T Vector<T>::getX() const
    {
        return this->x;
    }

    template<class T>
    T Vector<T>::getY() const
    {
        return this->y;
    }

    template<class T>
    T Vector<T>::getZ() const
    {
        return this->z;
    }

    template<class T>
    Vector<T> Vector<T>::productoEscalar(const T &s) const {
        return Vector<T>(s * this->x,
                         s * this->y,
                         s * this->z);
    }

    template<class T>
    T Vector<T>::productoPunto(const Vector<T> &v) const
    {
        return this->x * v.getX() + this->y * v.getY() + this->z * v.getZ();
    }

    template<class T>
    Vector<T> Vector<T>::productoCruz(const Vector<T> &v) const
    {
        return Vector<T>(this->y * v.getZ() - this->z * v.getY(),
                         this->z * v.getX() - this->x * v.getZ(),
                         this->x * v.getY() - this->y * v.getX());
    }

    template<class T>
    bool Vector<T>::operator==(const Vector<T> &v) const
    {
        return x == v.getX() && y == v.getY() && z == v.getZ();
    }

    template<class T>
    bool Vector<T>::operator!=(const Vector<T> &v) const
    {
        return x != v.getX() || y != v.getY() || z != v.getZ();
    }

    template<class T>
    Vector<T>& Vector<T>::operator=(const Vector<T> &v)
    {
        this->x = v.getX();
        this->y = v.getY();
        this->z = v.getZ();

        return *this;
    }

    template<class T>
    Vector<T> Vector<T>::operator+(const Vector<T> &v) const
    {
        return Vector<T>(this->x + v.getX(),
                         this->y + v.getY(),
                         this->z + v.getZ());
    }

    template<class T>
    Vector<T> Vector<T>::operator-() const
    {
        return Vector<T>(-x, -y, -z);
    }

    template<class T>
    Vector<T> Vector<T>::operator-(const Vector<T> &v) const
    {
        return Vector<T>(this->x - v.getX(),
                         this->y - v.getY(),
                         this->z - v.getZ());
    }

    template<class T>
    Vector<T> Vector<T>::operator*(const T &s) const
    {
        return productoEscalar(s);
    }

    template<class T>
    Vector<T> operator*(T s, Vector<T> &v)
    {
        return v.productoEscalar(s);
    }

    template<class T>
    T Vector<T>::operator*(const Vector<T> &v) const
    {
        return productoPunto(v);
    }

    template<class T>
    T Vector<T>::moduloCuadrado() const
    {
        return x*x + y*y + z*z;
    }

    template<class T>
    double Vector<T>::modulo() const
    {
        return sqrt(moduloCuadrado());
    }

    template<class T>
    Vector<double> Vector<T>::normalizado()
    {
        double modulo = this->modulo();
        return {((double) this->getX())/modulo,
                ((double) this->getY())/modulo,
                ((double) this->getZ())/modulo};
    }
}

#endif //TERRAINGEN_GEOMETRY_H
